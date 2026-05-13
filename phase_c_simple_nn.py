import numpy as np;

x= np.array([
    [2.0,3.0,4.0],
    [1.0,0.2,0.5],
    [-1.0,2.0,5.0]
]
)

# hidden layers (2 neurons) which take multi head attention means divided heads 
w1 = np.array([
    [0.1,0.5,-1.0],
    [0.5,-0.2,1.0]
])

b1 = np.array([1.0,0.5])

# z1 = np.dot(x,w1) + b1

z1 = x @ w1.T + b1

a1 = np.maximum(0,z1)

#output newuron 
w = np.array([1.0,-1.0])
b = 0.5

y = np.dot(a1,w) + b

print("Z1 : \n", z1)
print("A1 : \n", a1)
print("Output Y: \n", y)
