
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
        self.capacitor = capacity_factor

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




