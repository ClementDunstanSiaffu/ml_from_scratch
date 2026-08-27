
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


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



class MoeLayer(nn.Module):
    def __init__(self):
        super().__init__()
