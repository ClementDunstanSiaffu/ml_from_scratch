
import torch
import torch.nn as nn

class TransformerBloc(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim,num_heads,batch_first=True)
        self.laye_norms_1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
                     nn.Linear(embed_dim,4*embed_dim),
                     nn.GELU(),
                     nn.Linear(embed_dim*4,embed_dim)
        )
        self.laye_norms_2 = nn.LayerNorm(embed_dim)

    def forward(self,x):
        attent_out, w = self.attn(x,x,x)
        x = self.laye_norms_1(x + attent_out)
        mlp_out = self.mlp(x)
        x = self.laye_norms_2(x + mlp_out)
        return x

    


class GPTmini(nn.Module):
    def __init__(self,vocab_size,embed_dim,number_heads,num_layers,max_len=100):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size,embed_dim)
        self.position_embed = nn.Embedding(max_len,embed_dim)
        self.blocks = nn.Sequential(*[
            TransformerBloc(embed_dim,number_heads)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_dim,vocab_size)

    def forward(self,x):
        B, T = x.shape

        positions = torch.arange(0,T).unsqueeze(0)

        x = self.token_embed(x) + self.position_embed(positions)

        x = self.blocks(x)

        logits = self.fc_out(x)

        return logits


model = GPTmini(vocab_size=20,embed_dim=8,number_heads=2,num_layers=2)

tokens = torch.randint(0,20,(2,4))

out = model(tokens)

print("CHECKING OUT", out.shape,out)