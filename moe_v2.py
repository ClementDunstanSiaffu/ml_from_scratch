
import torch
import torch.nn as nn
import torch.nn.functional as F 


class SwiguExpertModel (nn.Module):

    def __init__(self,hidden_state,intermediate_state):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_state,intermediate_state,bias=False)
        self.up_proj = nn.Linear(hidden_state,intermediate_state,bias=False)
        self.down_proj = nn.Linear(intermediate_state,hidden_state,bias=False)

    def forward(self,x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = up * gate
        output = self.down_proj(hidden)
        return output
    


class MoeLayer (nn.Module):
    def __init__(self,hidden_state,intermediate_state,num_experts):
        super().__init__(self)
        self.hidden_state = hidden_state
        self.intermediate_state = intermediate_state
        self.num_experts = num_experts

        self.router = nn.Linear(hidden_state,num_experts,bias=False)

        self.experts = nn.ModuleList([
            SwiguExpertModel(hidden_state,intermediate_state)
            for _ in range (num_experts)
        ])

    def forward (self,x):
        batch_size,seq_len,hidden_state = x.shape

        tokens = x.view(-1,hidden_state)

        num_tokens = tokens.shape[0]

        router_logits = self.router(tokens)

        router_prob = F.softmax(router_logits,dim=-1)

        top1_prob,expert_ids = torch.max(router_prob,dim=-1)

        sorted_expert_ids,sort_idx = torch.sort(expert_ids)

        sorted_tokens = tokens[sort_idx]

        unique_experts_ids,counts = torch.unique_consecutive(sorted_expert_ids,returns_count=True)

        sorted_output = torch.empty_like(sorted_tokens)

        start = 0

        for expert_id,count in zip (
            unique_experts_ids,
            counts.toList()
        ):
            end = start + count

            expert_inputs = sorted_tokens[start:end]

            expert_outputs = self.experts[expert_id](expert_inputs)

            sorted_output = expert_outputs[start:end]

            start = end

        output_tokens = torch.empty_like(sorted_output)

        output_tokens[sort_idx] = sorted_output

        output = output_tokens.view(batch_size, seq_len, hidden_state)

        return output




        