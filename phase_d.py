
import torch;
import torch.nn as nn;
import torch.nn.functional as F;

class MiniTransformerBloc(nn.Module):
    def __init__(self,embed_dim,hidden_dim):
        super().__init__()

        #SELF ATTENTION
        self.attention = nn.MultiheadAttention(embed_dim,num_heads=1,batch_first=True)

        #LAYERS NORMS 
        self.norms1 = nn.LayerNorm(embed_dim)
        self.norms2 = nn.LayerNorm(embed_dim)

        #MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,embed_dim)
        )
    
    def forward(self,x):
        print("Input tokens :",x.shape)

        #SELF ATTENTION
        attn_output,attn_weights = self.attention(x,x,x)

        #LAYER NORMALAZATION
        x = self.norms1(x + attn_output)

        #FEED FOWARD LAYER
        mlp = self.mlp(x)

        #LAYER NORMALAZATION
        x = self.norms2(x + mlp)

        return x

x = torch.randn(2,4,8)
model = MiniTransformerBloc(embed_dim=8,hidden_dim=32)
output = model(x)

print("Final output :",output.shape)
print("Final output two :",output)
