# Time Spent (mins) | Pages Visited | Purchase Intent
# 0.90              | 0.85          | 1
# 0.70              | 0.60          | 1
# 0.30              | 0.20          | 0
# 0.40              | 0.25          | 0
# 0.80              | 0.75          | 1
# 0.35              | 0.40          | 0
# 0.95              | 0.90          | 1
# 0.50              | 0.45          | 0
# 0.85              | 0.80          | 1
# 0.60              | 0.55          | 1

import numpy as np

X = np.array([
    [0.90, 0.85],
    [0.70, 0.60],
    [0.30, 0.20],
    [0.40, 0.25],
    [0.80, 0.75],
    [0.35, 0.40],
    [0.95, 0.90],
    [0.50, 0.45],
    [0.85, 0.80],
    [0.60, 0.55],
])

Y = np.array([
    [1],
    [1],
    [0],
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
    [1],
])


np.random.seed(42)
w1 = np.random.rand()
w2 = np.random.rand()
w3 = np.random.rand()
w4 = np.random.rand()
w5 = np.random.rand()
w6 = np.random.rand()
b1 = np.random.rand()
b2 = np.random.rand()
b3 = np.random.rand()

lr = 0.1
epochs = 10000


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def binary_cross_entropy(y_true, y_pred):
    return -(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))


for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X)):
        x1, x2 = X[i]
        target = Y[i][0]

        z1 = w1 * x1 + w2 * x2 + b1
        a1 = sigmoid(z1)

        z2 = w3 * x1 + w4 * x2 + b2
        a2 = sigmoid(z2)

        z3 = w5 * a1 + w6 * a2 + b3
        output = sigmoid(z3)

        loss = binary_cross_entropy(target , output)
        total_loss += loss

        d_output = (output - target) * sigmoid_derivative(output)

        dw5 = d_output * a1
        dw6 = d_output * a2
        db3 = d_output

        d_a1 = d_output * w5 * sigmoid_derivative(a1)
        d_a2 = d_output * w6 * sigmoid_derivative(a2)

        dw1 = d_a1 * x1 
        dw2 = d_a1 * x2
        db1 = d_a1 

        dw3 = d_a2 * x1 
        dw4 = d_a2 * x2
        db2 = d_a2 
    
        #  Update weight and bias
        w1 -= lr * dw1
        w2 -= lr * dw2
        w3 -= lr * dw3
        w4 -= lr * dw4
        w5 -= lr * dw5
        w6 -= lr * dw6
        b1 -= lr * db1
        b2 -= lr * db2
        b3 -= lr * db3


    if epoch % 1000 == 0:
        print(f"Epochs {epoch} , Loss { total_loss:.4f}")


for i in range(len(X)):
    x1, x2 = X[i]

    z1 = w1 * x1 + w2 * x2 + b1
    a1 = sigmoid(z1)

    z2 = w3 * x1 + w4 * x2 + b2
    a2 = sigmoid(z2)

    z3 = w5 * a1 + w6 * a2 + b3
    output = sigmoid(z3)  

    prediction = 1 if output > 0.5 else 0
    print(f"Input : {X[i]} , Prediction value : {prediction} , Prob : {output :.4f}")  