
import torch
import torch.nn as nn
import torch.optim as optim


class TransformerBloc(nn.Module):
    def __init__(self,embed_dim,number_heads):
        super.__init__()
        self.multiHeadAttention = nn.MultiheadAttention(embed_dim,number_heads,batch_first=True)
        self.firstLayerNormalization = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim,4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim,embed_dim)
        )
        self.secondLayerNormalization = nn.LayerNorm(embed_dim)

    def forward(self,x):
        attention_out,w = self.multiHeadAttention(x,x,x)
        x = self.firstLayerNormalization(x + attention_out);
        mlp_out  = self.mlp(x)
        x = self.secondLayerNormalization(x + mlp_out)
        return x


class GPTMiniModel (nn.Module):
    def __init__(self,vocab_size,embed_dim,number_heads,num_layers,max_length = 100):
        super.__init__()
        self.token_embeddings = nn.Embedding(vocab_size,embed_dim)
        self.position_embeddings = nn.Embedding(max_length,embed_dim)
        self.blocks = nn.Sequential(*[
            TransformerBloc(embed_dim,number_heads)
            for _ in range(num_layers)
        ])
        self.logit_out = nn.Linear(embed_dim,vocab_size)
    
    def forward (self,x):
        B, T = x.shape

        positions = torch.arange(0,T).unsqueeze(0)

        x = self.token_embeddings(x) + self.position_embeddings(positions)

        x = self.blocks(x)

        logits = self.logit_out(x)

        return logits
    
    def generation ():
        