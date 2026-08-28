
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


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
    def __init__(self):
        super().__init__()
