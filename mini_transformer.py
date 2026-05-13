
import torch
import torch.nn as nn

vocab_size = 10
embed_dim = 8

embedding = nn.Embedding(vocab_size,embed_dim)

tokens = torch.tensor([[1,2,3,4,5,6,7,8,9]])

x = embedding(tokens)

print(x.shape)
print(x)

position_embedding = nn.Embedding(100,embed_dim)

positions = torch.arange(0,4).unsqueeze(0)

x = x + position_embedding(positions)