
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class MoeConfig:
    hidden_state:int = 16
    num_experts:int = 2
    top_k:int = 2
    intermediate_size:int = 32
    capacity_factor:float = 1.25
    router_aux_coeffient:float = 0.01


class SwiguExpertModel(nn.Module):

    def __init__(self,hidden_state,intermediate_state):
        self.gate = nn.Linear(hidden_state,intermediate_state)
        self.up = nn.Linear(hidden_state,intermediate_state)
        self.down = nn.Linear(intermediate_state,hidden_state)

    def forward(self,x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


@dataclass
class RouteMetadata:

    #This section will be used during the summation to get output. Eg context * probability of E1 + context * probabilty * E2
    #This produce Top 2 expert indices. Eg [[0,1],[2,4]] 
    top2_experts_indices:torch.tensor
    #This produce Top 2 expert probability. Eg [[0.6,0.7],[0.7,0.8]]
    top2_routing_probs:torch.tensor

    #This is used to simplify the access of the contents
    #This is flatten experts id, as expected experts ids is [[E1,E2]] => [E1,E2]
    flat_expert_ids:torch.tensor
    #This is used to flattening the tokens indices 
    flat_token_indices:torch.tensor
    #This is used to flattening the routing probs 
    flat_probs:torch.tensor

    #This return sorted items as it arrange all in order 
    sorted_expert_ids:torch.tensor
    sort_token_indices:torch.tensor
    sorted_probs:torch.tensor

    #This return accepted tokens and overflow tokens from experts 
    accepted_mask:torch.tensor
    overflow_mask:torch.tensor

    #As experts are unique so you need to get those unique expert and their respectively count 
    experts_count:torch.tensor
    experts_offset:torch.tensor

    #This used to return to their original position (re-order)
    original_position:torch.tensor

    capacity:int


class MoeLayer(nn.Module):

    def __init__(self,config:MoeConfig):
        super().__init__()
        self.hidden_state = config.hidden_state
        self.intermediate_state = config.intermediate_size
        self.top_k = config.top_k
        self.num_experts = config.num_experts
        self.router_aux_coeffient = config.router_aux_coeffient
        self.capacity_factor = config.capacity_factor

        if self.top_k != 2:
            raise ValueError("Moe v6 works for top k = 2")

        if self.top_k > self.num_experts:
            raise ValueError("The top k should not be greater than number of experts")

        self.router = nn.Linear(self.hidden_state,self.num_experts,bias=False)

        self.experts = nn.ModuleList(
            [
                SwiguExpertModel(self.hidden_state,self.intermediate_state)
                for _ in range (self.num_experts)
            ]
        )

    def route_tokens(self,tokens:torch.tensor):

        router_logits = self.router(tokens)

        routing_probs = F.softmax(router_logits,dim=-1)

        top2_experts,top2_probs = torch.topk(routing_probs,self.top_k,dim=-1)

        top2_probs = (top2_probs/ top2_probs.sum(dim=-1,keepdim=True))

        return (
            routing_probs,
            top2_probs,
            top2_experts
        )

    def build_metadata(self,top2_experts,top2_probs)->RouteMetadata:

        if self.top_k != 2 :
            raise ValueError("Moe v6 works for top k = 2")

        if self.top_k > self.num_experts:
            raise ValueError("The top k should not be greater than number of experts")

        num_tokens = top2_experts.size(0)

        flat_experts_id = (top2_experts.reshape(-1).long())

        flat_probs = top2_probs.reshape(-1)

        flat_token_indices = (
                                torch.arange(num_tokens,device=top2_experts.device)
                                .unsqueeze(1)
                                .expand(-1,self.top_k)
                                .reshape(-1)
                            )

        num_assignments = flat_experts_id.numel()

        capacity = max(
            1,
            int(
                torch.ceil(
                    torch.tensor(
                        (num_assignments/self.num_experts)*self.capacity_factor
                    )
                ).item()
            )

        )

        sorted_expert_ids,sort_order = torch.sort(flat_experts_id)

        sorted_token_indices_all = (flat_token_indices[sort_order])

        sorted_probs_all = flat_probs[sort_order]

        accepted_mask = torch.zeros(
                            num_assignments,
                            dtype=torch.bool,
                            device=flat_experts_id.device
        )

        expert_position = torch.zeros(
                            self.num_experts,
                            dtype=torch.long,
                            device=flat_experts_id.device
        )

        accepted_mask = torch.zeros(
                            num_assignments,
                            dtype=torch.bool,
                            device=flat_experts_id.device
        )

        experts_position = torch.zeros(
                            self.num_experts,
                            dtype=torch.long,
                            device=flat_experts_id.device
        )

        for sorted_position in range(num_assignments):

            expert_id = sorted_expert_ids[sorted_position]

            actual_position = expert_position[expert_id]

            if actual_position < capacity:
                original_expert_index = sort_order[actual_position]
                accepted_mask[original_expert_index] = True
            experts_position[actual_position]+=1

        overflow_mask = ~accepted_mask

        accepted_experts_id = flat_experts_id[accepted_mask]

        accepted_token_indices = flat_token_indices[accepted_mask]

        accepted_route_prob = flat_probs[accepted_mask]

        if accepted_experts_id.numel() > 0 :

            sorted_expert_ids_accepted, sorted_expert_order = torch.sort(accepted_experts_id)

            sort_token_indices_accepted = accepted_token_indices[sorted_expert_order]

            sorted_probs_accepted = accepted_route_prob[sorted_expert_order]

        else:

            sorted_expert_ids_accepted = torch.empty(
                                    0,
                                    dtype=torch.long,
                                    device=flat_experts_id.device
            )

            sort_token_indices_accepted = torch.empty(
                                    0,
                                    dtype=torch.long,
                                    device=flat_token_indices.device

            )

            sorted_probs_accepted = torch.empty(
                                0,
                                dtype=torch.long,
                                device=flat_probs.device
            )

        experts_count = torch.bincount(sorted_expert_ids_accepted,minlength=self.num_experts)

        experts_offset = torch.empty(
                            0,
                            dtype=torch.long,
                            device=flat_experts_id.device
        )

        if self.num_experts > 0:
            experts_offset[1:] = torch.cumsum(
                experts_count[:1],
                dim=0
            )

        return RouteMetadata(
            top2_experts_indices=top2_experts,
            top2_routing_probs=top2_probs,
            flat_expert_ids=flat_experts_id,
            flat_probs=flat_probs,
            capacity=capacity,
            sorted_expert_ids=sorted_expert_ids_accepted,
            sorted_probs=sorted_probs_accepted,
            overflow_mask=overflow_mask,
            accepted_mask=accepted_mask,
            sort_token_indices=sort_token_indices_accepted,
            experts_count=experts_count,
            experts_offset=experts_offset,
            flat_token_indices=flat_token_indices
        )


    def dispatch(self,tokens:torch.tensor,metadata:RouteMetadata):

        if metadata.sort_token_indices == 0:
            return torch.new_empty(0,self.hidden_state)

        return tokens[metadata.sort_token_indices]

    def execute_experts(self,dispatched_tokens:torch.tensor,metadata:RouteMetadata):

        experts_output = dispatched_tokens.new_zeros(dispatched_tokens.shape)

        for expert_id in range(self.num_experts):

            count = metadata.experts_count[expert_id].item()

            if count == 0:
                continue

            start = metadata.experts_offset[expert_id].item()

            end = start + count

            ouput = self.experts[expert_id](dispatched_tokens[start:end])

            experts_output[start:end] = ouput

        return experts_output

    def restore_tokens(self,experts_output:torch.tensor,metadata:RouteMetadata,num_tokens):

        output = experts_output.new_zeros((num_tokens,self.hidden_state))

        if metadata.sort_token_indices == 0:
            return output

        weighted_output = (experts_output * metadata.sorted_probs.unsqueeze(-1))

        output = torch.index_add(
                    0,
                    metadata.sort_token_indices,
                    weighted_output
        )

        return output

    # def load_balancing(self):


        


            

