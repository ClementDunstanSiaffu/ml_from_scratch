import torch 
import torch.nn as nn
import torch.nn.functional as F


class SwiguExpertModel (nn.Module):

    def __init__(self,hidden_state,intermediate_size):
        super().__init__()
        self.gate_proj  = nn.Linear(hidden_state,intermediate_size,bias=False)
        self.up_proj = nn.Linear(hidden_state,intermediate_size,bias=False)
        self.down_proj = nn.Linear(intermediate_size,hidden_state,bias=False )

    def forward(self,x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        output = self.down_proj(hidden)
        return output
    


class MoELayer (nn.Module):

    def __init__(self,hidden_state,intermediate_state,num_experts,top_k:int=2):
        super().__init__()

        self.hidden_state = hidden_state
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(hidden_state,num_experts,bias=False)

        self.experts = nn.ModuleList(
            [
                SwiguExpertModel(hidden_state,intermediate_state)
                for _ in range(num_experts)
            ]
        )

    def forward(self,x):

        batch_num,sequence_len,hidden_size = x.shape

        tokens = x.view(-1,hidden_size)

        numb_tokens = tokens.shape[0]

        router_logits = self.router(tokens)

        router_probs = F.softmax(router_logits, dim=-1)

        topk_probs,topk_indices = torch.topk(router_probs,k=self.top_k,dim = -1)

        combined_output = torch.zeros_like(tokens)

        for experts_id in range (self.num_experts):
            token_positions = []
            token_weights = []

            for token_id in range (numb_tokens):

                for router_id in range (self.top_k):
                    if (topk_indices[token_id,router_id].item() == experts_id):
                        token_positions.append(token_id)
                        token_weights.append(topk_probs[token_id,router_id])

            if len(token_positions) == 0:
                continue

            token_positions = torch.tensor(token_positions,device= x.device)

            token_weights = torch.stack(token_weights)

            experts_inputs = tokens[token_positions]

            expert_output = self.experts[experts_id](experts_inputs)

            expert_output = (expert_output * token_weights.unsqueeze(-1))

            combined_output[token_positions]+= expert_output

        output = combined_output.view(
            batch_num,
            sequence_len,
            hidden_size
        )

        return output



def main ():
    torch.manual_seed(42)

    batch_size = 4
    sequence_length = 2
    hidden_size = 8
    intermediate_size = 16

    num_experts = 4
    top_k = 2

    moe = MoELayer(hidden_state=hidden_size,intermediate_state=intermediate_size,num_experts=num_experts,top_k=top_k)

    x = torch.randn(batch_size,sequence_length,hidden_size)

    print("CHECKING THE SHAPE",x.shape)

    print ("\nCHECKING THE ACTUAL DATA",x)

    output = moe(x)

    print("\nCHECKING THE MOE SHAPE",output.shape)
    print("\nCHECKING THE MOE SHAPE ACTUAL DATA",output)


if (__name__ == "__main__"):
    main()




    