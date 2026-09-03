
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


#CONFIGURATION CLASS 
@dataclass
class MoEConfig :
    hidden_state:int = 16
    intermediate_state:int = 32
    num_experts:int = 4

    capacity_factor:float = 1.25
    router_aux_loss_coef:float = 0.01



class SwiguExpertModel(nn.Module):
    def __init__(self,hidden_state,intermediate_state):
        super().__init__()

        self.gate = nn.Linear(hidden_state,intermediate_state)
        self.up = nn.Linear(hidden_state,intermediate_state)
        self.down = nn.Linear(intermediate_state,hidden_state)

    def forward(self,x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


@dataclass
class RoutingMetadata:

    expert_indices:torch.tensor
    routing_probs:torch.tensor

    accepted_mask:torch.tensor
    overflow_mask:torch.tensor

    sorted_indices:torch.tensor

    sorted_expert_ids:torch.tensor

    unique_experts:torch.tensor
    expert_counts:torch.tensor

    original_positions:torch.tensor


class MoeLayer(nn.Module):
    def __init__(self,config:MoEConfig):
        super().__init__()

        self.hidden_state = config.hidden_state
        self.intermediate_state = config.intermediate_state
        self.num_experts  = config.num_experts
        self.capacity_factor = config.capacity_factor
        self.router_aux_loss_coef = config.router_aux_loss_coef

        self.router = nn.Linear(self.hidden_state,self.num_experts,bias=False)

        self.experts = nn.ModuleList([
            SwiguExpertModel(self.hidden_state,self.intermediate_state)
            for _ in range(self.num_experts)
        ])

    #It is used to route token to the experts, so it decide which expert take which token
    def route_tokens(self,tokens:torch.tensor):

        # Get logits, logits means which has more likely to be selected compared to other but still it not in probability so it needs softmax
        router_logits = self.router(tokens)

        #Softmax the logits to get the probability, take probability across tokens (dim=-1)
        router_probs = F.softmax(router_logits,dim=-1)

        #From the router_probs across the tokens get the max probability and experts id for the expert for each token with top 1 
        top1_probs,top1_experts = torch.max(router_probs,dim=-1)

        #Get each experts probability on each expert column
        probability_fraction = router_probs.mean(dim=0)

        #Find routing fraction, token count
        token_count = torch.bincount(router_probs,minlength=self.num_experts)

        routing_fraction  = (token_count.float()/tokens.size(0))

        #Find the auxially loss which is done during the training for balance token on each experts 
        aux_loss = self.num_experts * torch.sum(probability_fraction * routing_fraction)

        aux_loss = self.router_aux_loss_coef * aux_loss

        return (
            top1_probs,
            top1_experts,
            aux_loss
        )

    def buildMetadata(self,expert_indices:torch.tensor,routing_probs:torch.tensor)->RoutingMetadata:

        num_tokens = expert_indices.size[0]

        expert_capacity = int(num_tokens * self.capacity_factor)/self.num_experts

        expert_capacity = max(expert_capacity,1)

        # This hold the spot on the expert. Example in the Expert0 (0,1,2) means this token will take this position (slot) in the expert
        token_positions = torch.zeros_like(expert_indices)

        # This loops help to get token_positions which define spot on the Expert
        # The aim of the token_positions will be used to get the accepted mask as token position as it provide the slot for those token on the specific expert
        # So for example the slot goes to 5 but maximum capacity is 4 so the slot 5 for the expert will be overflow so for the token which occupy that slot will overflow  
        for expert_id in range(self.num_experts):

            expert_mask = expert_indices == expert_id

            positions = torch.cumsum(expert_mask.long(),dim=0) - 1

            token_positions = torch.where(expert_mask,positions,token_positions)

        # Get all the accepted mask which return array of boolean [True,True] for those accepted token spot (slot) for that expert
        accepted_mask = token_positions < expert_capacity

        # All negation of the accepted mask becomes overflow_mask 
        overflow_mask = ~accepted_mask

        # It returns index of all accepted mask it inputs [True,True] => [0,1]
        # The aim of the accepted positions as since it returns index of the accepted condition using accepted mask which accumulated positions are accepted so we can use their indices from the general expert_indices to get accepted expert ids 
        # [1,2,3,4,5]
        accepted_positions = torch.nonzero(accepted_mask,as_tuple=False).unsqueeze(-1)

        # Now we need to get using the accepted positions and that is the reason why we needed accepted_positions
        # Now we have all the accepted expert ids 
        # Let say we got this [0,0,1,2,0]
        accepted_expert_ids = expert_indices[accepted_positions]

        # Now we need to sort by this sort it sorts using the index. Example from the accepted experts id we got [0,0,1,2,0] then => it will be [0,1,4,2,3]
        # As it takes index of the order 
        # Why we need sort order ? because it will be used to get sort indices and sorted expert id so it provide one sorting for sort indices and sorted expert id 
        # Let say we got this [0,1,4,2,3]
        sort_order = torch.argsort(accepted_expert_ids)

        # As the accepted_positions contains all accepted experts ids in order means [0,1,2,3,4] without know which is the slot for the which expert or which is from this expert 
        # Using sort order from accepted position we get [1,2,4,2,3]
        # This will help to get tokens embeddings for the specific expert 
        sort_indices = accepted_positions[sort_order]

        #[0,0,0,1,2]
        # This will help to get expert id and counts on how many times did experts occur 
        sorted_expert_ids = accepted_expert_ids[sort_order]

        if sorted_expert_ids.numel() > 0 :
            unique_expert,expert_counts = torch.unique_consecutive(sorted_expert_ids,return_counts=True)
        else:
            unique_expert = torch.empty(0,device=torch.long,device=expert_indices.device)
            expert_counts = torch.empty(0,dtype=torch.long,device=expert_indices.device)

        original_positions = sort_indices.clone()

        return RoutingMetadata(
            expert_indices=expert_indices,
            routing_probs=routing_probs,
            accepted_mask=accepted_mask,
            overflow_mask=overflow_mask,
            sorted_index=sort_indices,
            sorted_expert_ids=sorted_expert_ids,
            unique_experts=unique_expert,
            expert_counts=expert_counts,
            original_positions=original_positions
        )

    def dispatch(self,tokens:torch.tensor,metadata:RoutingMetadata):
        return tokens[metadata.sorted_indices]

    def execute_experts (self,dispatched_token:torch.tensor,metadata:RoutingMetadata):

        start = 0

        output  = torch.empty_like(dispatched_token)

        for expert_id,count in zip (metadata.unique_experts.tolist(),metadata.expert_counts.tolist()):
            end = start + count

            expert_tokens = dispatched_token[start:end]

            expert_output = self.experts[expert_id](expert_tokens)

            output[start:end] = expert_output

            start = end

        return output













         









