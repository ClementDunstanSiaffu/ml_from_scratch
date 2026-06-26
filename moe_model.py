
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(42)

num_tokens = 3
hidden_state = 4
num_experts = 3
top_k = 2
num_ffns = 8

H = torch.randn(num_tokens,hidden_state)

print(H.shape)
print(H)


class Expert (nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(
            hidden_state,
            num_ffns
        )

        self.fc2 = nn.Linear(
            num_ffns,
            hidden_state
        )

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x    