# GPA | Experience | Extra | Salary (normalized)
# 0.9 | 0.8        | 0.5   | 0.9
# 0.6 | 0.5        | 0.3   | 0.65
# 0.4 | 0.2        | 0.1   | 0.4
# 0.7 | 0.9        | 0.6   | 0.85

import numpy as np


def count_weights_and_biases(layers):   
    """
    Parameters:
    layers (list): List of integers where each element is the number of neurons
                   in that layer. Includes input, hidden, and output layers.

    Returns:
    (int, int): Total weights and total biases in the MLP.
    """
    total_weights = 0
    total_bias = 0

    for i in range(1, len(layers)):
        total_weights += layers[i - 1] * layers[i]
        total_bias += layers[i]
    
    return total_weights, total_bias

# Architecture: Input=3, Hidden1=2, Output=1
architecture = [3, 2, 1]

weights, biases = count_weights_and_biases(architecture)
print("Total Weights:", weights)
print("Total Biases:", biases)

def sigmoid(x):
    return 1/ (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([
    [0.9, 0.8, 0.5],
    [0.6, 0.5, 0.3],
    [0.4, 0.2, 0.1],
    [0.7, 0.9, 0.6]
])
Y = np.array([
    [0.9],
    [0.65],
    [0.4],
    [0.85]
])

np.random.seed(42)
w1 = np.random.rand()
w2 = np.random.rand()
w3 = np.random.rand()
w4 = np.random.rand()
w5 = np.random.rand()
w6 = np.random.rand()
w7 = np.random.rand()
w8 = np.random.rand()
b1 = np.random.rand()
b2 = np.random.rand()
b3 = np.random.rand()

lr = 0.1
epochs = 10000

for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X)):
        x1, x2, x3 = X[i]
        target = Y[i][0]

        # Forward Pass

        z1 = w1 * x1 + w2 * x2 + w3 * x3 + b1
        a1 = sigmoid(z1)

        z2 = w4 * x1 + w5 * x2 + w6 * x3 + b2
        a2 = sigmoid(z2) 

        z3 = w7 * a1 + w8 * a2 + b3
        output = sigmoid(z3)

        # loss
        loss = (target - output) ** 2
        total_loss += loss

        # Backpropapagation
        d_output = 2 * (output - target) * sigmoid_derivative(output)

        # Gradients for output weights and bias
        dw7 = d_output * a1
        dw8 = d_output * a2
        db3 = d_output

        # Hidden layer errors
        d_a1 = d_output * w7 * sigmoid_derivative(a1)
        d_a2 = d_output * w8 * sigmoid_derivative(a2)

        # Gradients for hidden weights and biases
        dw1 = d_a1 * x1
        dw2 = d_a1 * x2
        dw3 = d_a1 * x3
        db1 = d_a1

        dw4 = d_a2 * x1
        dw5 = d_a2 * x2
        dw6 = d_a2 * x3
        db2 = d_a2

        # Update weights and biases
        w1 -= lr * dw1
        w2 -= lr * dw2
        w3 -= lr * dw3
        b1 -= lr * db1    

        w4 -= lr * dw4
        w5 -= lr * dw5
        w6 -= lr * dw6
        b2 -= lr * db2

        w7 -= lr * dw7
        w8 -= lr * dw8
        b3 -= lr * db3
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")



print("\nPredictions:")
for i in range(len(X)):
    x1, x2, x3 = X[i]    

    z1 = w1 * x1 + w2 * x2 + w3 * x3 + b1
    a1 = sigmoid(z1)

    z2 = w4 * x1 + w5 * x2 + w6 * x3 + b2
    a2 = sigmoid(z2) 

    z3 = w7 * a1 + w8 * a2 + b3
    output = sigmoid(z3)

    print(f"Input: {X[i]}, Predicted Salary: {output * 100000:.2f}")
