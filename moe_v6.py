
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

    top2_experts_indices:torch.tensor
    top2_routing_probs:torch.tensor

    flat_expert_ids:torch.tensor
    flat_token_indices:torch.tensor
    flat_probs:torch.tensor


    sorted_expert_ids:torch.tensor
    sort_token_indices:torch.tensor
    sorted_probs:torch.tensor

    accepted_mask:torch.tensor
    overflow_mask:torch.tensor

    experts_count:torch.tensor
    experts_offset:torch.tensor

    original_position:torch.tensor

    capacity:int

