
import numpy as np;

def softmax (x):
    exp_x = np.exp(x-np.max(x,axis = -1,keepdims=True))
    return exp_x / np.sum(exp_x,axis=1,keepdims=True)

X = np.random.randn(2,4,8)

Wq = np.random.randn(8,8)
Wk = np.random.randn(8,8)
Wv = np.random.randn(8,8)


Q = X @ Wq
K = X @ Wk
V = X @ Wv

scores = Q @ K.transpose(0,2,1)

d = Q.shape[-1]

scores = scores / np.sqrt(d)

weights = softmax(scores)

context = weights @ V

print("Q shape : ",Q.shape,Q)
print("Score shape :",scores.shape,scores)
print("weights shape :",weights.shape,weights)
print("context shape : ",context.shape,context)