
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

        

        return RouteMetadata(
            top2_experts_indices=top2_experts,
            top2_routing_probs=top2_probs
        )

