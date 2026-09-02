
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

    def buildMetadata(self,expert_indices:torch.tensor,routing_probs:torch.tensor):

        num_tokens = expert_indices.size[0]

        expert_capacity = int(num_tokens * self.capacity_factor)/self.num_experts

        expert_capacity = max(expert_capacity,1)

        token_positions = torch.zeros_like(expert_indices)

        for expert_id in range(self.num_experts):

            expert_mask = expert_indices == expert_id

            positions = torch.cumsum(expert_mask.long(),dim=0) - 1

            token_positions = torch.where(expert_mask,positions,token_positions)


        accepted_mask = token_positions < expert_capacity

        overflow_mask = ~accepted_mask

        accepted_positions = torch.nonzero(accepted_mask,as_tuple=False).unsqueeze(-1)

        accepted_expert_ids = expert_indices[accepted_positions]

        sort_order = torch.argsort(accepted_expert_ids)

        sort_indices = accepted_positions[sort_order]

        sorted_expert_ids = accepted_expert_ids[sort_order]










         









