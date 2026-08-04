
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiguExpertModel(nn.Module):
    def __init__(self,hidden_state,intermediate_state):
        super().__init__()

        self.gate = nn.Linear(hidden_state,intermediate_state,bias=False)
        self.up_proj = nn.Linear(hidden_state,intermediate_state,bias=False)
        self.down_proj = nn.Linear(intermediate_state,hidden_state,bias=False)


    def forward(self,x):
        return self.down_proj(F.silu(self.gate(x))*self.up_proj(x))


class MoeExpertLayer(nn.Module):
    def __init__(self,hidden_state,intermediate_state,num_experts,capacity_factor):
        super().__init__()

        self.hidden_state = hidden_state
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        self.router = nn.Linear(hidden_state,num_experts)

        self.experts = nn.ModuleList(
            [
                SwiguExpertModel(hidden_state,intermediate_state)
                for _ in range (num_experts)
            ]
        )


    def compute_aux_loss (self,routing_probs,experts_ids):

        # GET THE EXPERT COLUMN AVERAGE ATTENTION, LIKE HOW MUCH EACH EXPERT GOT ATTENTION
        router_prob = routing_probs.mean(dim=0)

        # GET TOKENS COUNT FROM EACH EXPERTS 
        expert_counts = torch.bincount(experts_ids,minlength=self.num_experts).float()

        # GET THE PROBABILITY FOR EXPERTS COUNT (EXPERT FRACTION)
        expert_fraction = (expert_counts/experts_ids.numel())

        # FIND THE AUTH LOSS BY TAKING NUMBER OF EXPERTS TIMES THE SUM OF EXPERT ATTENTION TIMES EXPERTS FRACTION 
        aux_loss = self.num_experts*torch.sum(router_prob * expert_fraction)

        return aux_loss


    def forward(self,x):

        batch_size,seq_len, hidden_state = x.shape

        tokens = x.view(-1,hidden_state)

        num_tokens = tokens.shape[0]

        router_logits = self.router(tokens)

        router_probs = F.softmax(router_logits,dim=-1)

        top1_prob,experts_id = torch.max(router_probs,dim=-1)

        aux_loss = self.compute_aux_loss(top1_prob,experts_id)

        sorted_experts_id,sort_idx = torch.sort(experts_id)

        sorted_tokens = tokens[sort_idx]

        unique_expert_ids,counts = torch.unique_consecutive(sorted_experts_id,return_counts=True)

        avg_tokens = (num_tokens + self.num_experts - 1)//self.num_experts

        capacity = int (avg_tokens * self.capacity_factor)

        sorted_ouput = torch.empty_like(sorted_tokens)

        start = 0

        for expert_id,count in zip(unique_expert_ids,counts.tolist()):

            end = start + count

            experts_input = sorted_tokens[start:end]

            if (count <= capacity):

                experts_output = self.experts[expert_id](experts_input)

                sorted_ouput[start:end] = experts_output
            else:

                valid_inputs = experts_input[:capacity]

                overflow_inputs = experts_input[capacity:]

                valid_outputs = self.experts[expert_id](valid_inputs)

                sorted_ouput[start:start + capacity] = valid_outputs

                sorted_ouput[start + capacity:end] = overflow_inputs

            start = end

        output_tokens = torch.empty_like(sorted_ouput)

        output_tokens[sort_idx] = sorted_ouput

        output = output_tokens.view(batch_size,seq_len,hidden_state)

        return output,aux_loss



def main ():

    torch.manual_seed(42)

    hidden_state = 8
    intermediate_state = 16
    batch_size = 2
    seq_length = 4
    num_experts = 4

    moe = MoeExpertLayer(
        hidden_state=hidden_state,
        intermediate_state=intermediate_state,
        num_experts=num_experts,
        capacity_factor=1.25
    )

    x = torch.randn(batch_size,seq_length,hidden_state)

    output,aux_loss = moe(x)

    print("Input Shape")

    print(x.shape)

    print()

    print("Output Shape")

    print(output.shape)

    print()

    print("Auxiliary Loss")

    print(aux_loss.item())


if (__name__ == "__main__"):
    main()








