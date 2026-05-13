
import torch
import torch.nn.functional as F


X = torch.randn(2,4,8)

Wq = torch.randn(8,8)
Wk = torch.randn(8,8)
Wv = torch.randn(8,8)

Q = X @ Wq
K = X @ Wk
V = X @ Wv

scores = Q @ K.transpose(-2,-1)

d = Q.size(-1)

scores = scores/(d**0.5)

weights = F.softmax(scores,dim=1)

context = weights @ V

print("Context shape ",context.shape,context)

