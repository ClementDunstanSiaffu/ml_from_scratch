
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiguExpertModel(nn.Module):
    def __init__(self,hidden_state,intermediate_state):
        super().__init__()

        self.gate = nn.Linear(hidden_state,intermediate_state,bias=False)
        self.up = nn.Linear(hidden_state,intermediate_state,bias=False)
        self.down = nn.Linear(intermediate_state,hidden_state,bias=False)

    def forward(self,x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Router(nn.Module):
    def __init__(self,hidden_state,num_experts):
        super().__init__()

        self.linear = nn.Linear(hidden_state,num_experts,bias=False)

    def forward(self,x):
        logits = self.linear(x)
        probs = F.softmax(logits, dim=-1)
        weights,ids = torch.max(probs,dim=-1)
        return weights,ids,probs


class MoeLayer(nn.Module):
    def __init__(self,hidden_state=16,intermediate_state=32,num_experts=4,capacity_factor=1.25):
        super().__init__()

        self.hidden_state = hidden_state
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        self.router = Router(hidden_state, num_experts)
        self.experts = nn.ModuleList([
            SwiguExpertModel(hidden_state,intermediate_state)
            for _ in range (num_experts)
        ])

    def forward(self,tokens):

        num_tokens = tokens.size(0)

        router_weight,expert_ids,probs = self.router(tokens)
        sorted_expert_ids,sort_ids = torch.sort(expert_ids)
        sorted_tokens = tokens[sort_ids]
        sorted_weights = router_weight[sort_ids]
        unique_expert_ids,counts = torch.unique_consecutive(sorted_expert_ids,return_counts=True)
        sorted_outputs = torch.zeros_like(sorted_tokens)
        overflowed_mask_tokens = torch.zeros(num_tokens,dtype = torch.bool,device=tokens.device)

        avg = math.ceil(num_tokens/self.num_experts)
        experts_capacity = int(avg * self.capacity_factor)

        accepted_counts = []
        overflow_counts = []

        start = 0

        for expert,count in zip (unique_expert_ids.tolist(),counts.tolist()):

            accepted = min(count,experts_capacity)
            overflow = count - accepted
            accepted_counts.append(accepted)
            overflow_counts.append(overflow)

            if accepted > 0 :
                end = start + accepted
                out = self.experts[expert](sorted_tokens[start:end])
                out = out * sorted_weights[start:end].unsqueeze(-1)
                sorted_outputs[start:end] = out

            if overflow > 0:
                ov_start = start + accepted
                ov_end = start + count
                overflowed_mask_tokens[ov_start:ov_end] = True
                sorted_outputs[ov_start:ov_end] = sorted_tokens[ov_start:ov_end]

            start += count 

        outputs = torch.zeros_like(tokens)
        outputs[sort_ids] = sorted_outputs

        overflow_mask = torch.zeros_like(overflowed_mask_tokens)
        overflow_mask[sort_ids] = overflowed_mask_tokens

        stats = {
             "capacity": experts_capacity,
            "unique_experts": unique_expert_ids.tolist(),
            "counts": counts.tolist(),
            "accepted_counts": accepted_counts,
            "overflow_counts": overflow_counts,
            "overflow_tokens": int(sum(overflow_counts)),
            "overflow_mask": overflow_mask,
            "weight_probabilities": probs
        }

        return outputs,stats



def main():
    torch.manual_seed(42)

    x = torch.randn(20,512)

    moe = MoeLayer(hidden_state=512,intermediate_state=1024,num_experts=32)

    y,stats = moe(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)
    print("Capacity:", stats["capacity"])
    print("Experts:", stats["unique_experts"])
    print("Counts:", stats["counts"])
    print("Accepted:", stats["accepted_counts"])
    print("Overflow:", stats["overflow_counts"])
    print("Total overflow:", stats["overflow_tokens"])
    print("Checking the router probabilities",stats["weight_probabilities"])
    print("Checking the input",x)
    print("Checking the output",y)


if __name__ == "__main__":
    main()


