
x = 2.0

w = 1.0

b = 1.0

y_true = 10

y_pred = w*x + b

error = y_pred - y_true

print("Prediction : ",y_pred)
print("Error :", error)

loss = error ** 2

print("Loss :",loss)