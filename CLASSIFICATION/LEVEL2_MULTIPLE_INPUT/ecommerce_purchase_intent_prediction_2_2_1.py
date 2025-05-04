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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)

def binary_cross_entropy(y_true, y_pred):
    return -(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))

input_size = X.shape[1]
hidden_size = 2
output_size = 1

lr = 0.1
epochs = 10000

np.random.seed(42)
w1 = np.random.rand(input_size, hidden_size)
b1 = np.random.rand(hidden_size)
w2 = np.random.rand(hidden_size, output_size)
b2 = np.random.rand(output_size)

for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X)):
        x = X[i]
        target = Y[i][0]

        z1 = np.dot(x, w1) + b1
        a1 = sigmoid(z1)

        z2 = np.dot(x, w2) + b2
        output = sigmoid(z2)[0]

        loss = binary_cross_entropy(target, output)
        total_loss += loss


        d_output = (output - target) * sigmoid_derivative(output)
        dw2 = np.outer(a1, np.array([d_output]))
        db2 = np.array([d_output])

        d_hidden = np.dot(w2, np.array([d_output])) * sigmoid_derivative(a1)
        dw1 = np.outer(x, d_hidden)
        db1 = d_hidden

        w2 -= lr * dw2
        b2 -= lr * db2
        w1 -= lr * dw1
        b1 -= lr * db1

    if epoch % 1000 == 0:
        print(f"Total Epochs {epoch} , Total Loss: {total_loss:.4f}")


print("\nPredictions")
for i in range(len(X)):
    x = X[i]

    z1 = np.dot(x, w1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(x, w2) + b2
    output = sigmoid(z2)[0]

    prediction = 1 if output > 0.5 else 0
    print(f"Input {x}, predicted {prediction} , Prob {output:.4f}")

