
import numpy as np

# x = [2,3,4]
# w = [1,0.5,-1]
# b = 2

x = np.array([2,3,4])
w = np.array([1,0.5,-1])
b = 2

y = np.dot(w,x) + b

# y = 0

# for xi,wi in zip(x,w):
#     y+= xi * wi
# y += b

print("output",y)