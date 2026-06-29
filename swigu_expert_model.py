

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



model = SwigexpertModel(8,16)

inputs = torch.randint(0,20,(10,8)).float()

output = model(inputs)

print(output.shape)
print(output)