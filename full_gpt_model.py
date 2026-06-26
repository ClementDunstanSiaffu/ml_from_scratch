
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBloc(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super().__init__()

        # pre normalization
        self.pre_norms = nn.LayerNorm(embed_dim)

        #multi-head attention
        self.multi_head_attention = nn.MultiheadAttention(embed_dim,num_heads,batch_first=True)

        #second layer normalization
        self.second_layernorms = nn.LayerNorm(embed_dim)

        #MLP (Feed Fowards Network)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim,4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim,embed_dim)
        )

    def forward (self,x):
        norm_x = self.pre_norms(x)
        B,T, C = x.shape
        mask = torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()
        attent_x,_ = self.multi_head_attention(norm_x,norm_x,norm_x,attn_mask = mask)
        x = x+attent_x
        norm_x = self.second_layernorms(x)
        mlp_x =  self.mlp(norm_x)
        return mlp_x + x
    

class GPTmini (nn.Module):
    def __init__(self,vocabulary_size,embed_dim,num_heads,num_layers,max_length=100):
        super().__init__()

        # Create table for token embeddings 
        self.token_embed = nn.Embedding(vocabulary_size,embed_dim)

        # Create table for positional embeddings 
        self.position_embed = nn.Embedding(max_length,embed_dim)

        # Loop to the multiple Transformer bloc 
        self.blocks = nn.Sequential(*[
            TransformerBloc(embed_dim,num_heads)
            for _ in range (num_layers)
        ])
        
        # Layer normalization after 
        self.last_layer_norms = nn.LayerNorm(embed_dim)

        ## Set matrix for logits 
        self.logits_table = nn.Linear(embed_dim,vocabulary_size)

    def forward (self,x):

        # Getting the the number of tokens (T) and batch (B)
        B, T = x.shape

        # Then use T (token length) to determine the index (ids) for positions  
        positions = torch.arange(0,T).unsqueeze(0)

        # Add position embeddings to the token embeddings 
        x = self.token_embed(x) + self.position_embed(positions)

        # Loop that token inputs to the layers (Transformer bloc)
        x = self.blocks(x)

        # Normalize the token embeddings 
        x = self.last_layer_norms(x)

        # Get logits
        logits = self.logits_table(x)

        return logits 
    

model = GPTmini(50,18,6,4,100)

inputs_id = torch.randint(0,50,(10,100))

logits = model(inputs_id)

print(f"Checking the logit shape and the logits emebddings  : {logits.shape}, {logits}")





