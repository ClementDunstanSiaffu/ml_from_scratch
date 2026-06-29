

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwigexpertModel (nn.Module):

    def __init__(self,hidden_state,intermediate_size):
        super().__init__()

        self.gate_proj = nn.Linear(
            hidden_state,
            intermediate_size,
            bias=False
        )

        self.up_proj = nn.Linear(
            hidden_state,
            intermediate_size,
            bias=False
        )

        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_state,
            bias=False
        )

    def forward(self,x):
        gate = F.silu(
            self.gate_proj(x)
        )

        up = self.up_proj(x)

        hidden = gate * up

        output = self.down_proj(hidden)

        return output


class MoeLayer (nn.Module):
    def __init__(self,hidden_state,intermediate_state,num_experts,top_k):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(hidden_state,num_experts,bias=False)

        self.experts = nn.ModuleList(
            [
                SwigexpertModel(hidden_state, intermediate_state)
                for _ in range(num_experts)
            ]
        )