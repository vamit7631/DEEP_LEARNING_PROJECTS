# GPA | Experience | Placement (normalized)
# 0.9 | 0.8        | 1
# 0.6 | 0.5        | 1
# 0.4 | 0.2        | 0
# 0.7 | 0.9        | 0

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

# Architecture: Input=2, Hidden1=2, Output=1
architecture = [2, 2, 1]

weights, biases = count_weights_and_biases(architecture)
print("Total Weights:", weights)
print("Total Biases:", biases)

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):  # assuming x is sigmoid(x)
    return x * (1 - x)

def binary_cross_entropy(y_true, y_pred): # loss function
    return -(y_true * np.log(y_pred + 1e-9) +  (1 - y_true) * np.log(1 - y_pred + 1e-9))


# Input features: [GPA, Experience]
X = np.array([
    [0.9, 0.8],
    [0.6, 0.5],
    [0.4, 0.2],
    [0.7, 0.9]
])

# Output: expected salary (normalized between 0 and 1)
y = np.array([
    [1],
    [1],
    [0],
    [0]
])

# Initialize weights and biases randomly
np.random.seed(42)
w1 = np.random.rand()
w2 = np.random.rand()
w3 = np.random.rand()
w4 = np.random.rand()
b1 = np.random.rand()
b2 = np.random.rand()

w5 = np.random.rand()
w6 = np.random.rand()
b3 = np.random.rand()

lr = 0.1
epochs = 10000

# Training
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X)):
        x1, x2 = X[i]
        target = y[i][0]

        # Forward pass
        z1 = x1 * w1 + x2 * w2 + b1
        a1 = sigmoid(z1)

        z2 = x1 * w3 + x2 * w4 + b2
        a2 = sigmoid(z2)

        z3 = a1 * w5 + a2 * w6 + b3
        output = sigmoid(z3)

        # Compute loss
        loss = binary_cross_entropy(target, output)   # binary cross entropy loss function
        total_loss += loss

        # Backpropagation
        d_output = (output - target) * sigmoid_derivative(output)

        # Gradients for output layer
        dw5 = d_output * a1
        dw6 = d_output * a2
        db3 = d_output

        # Gradients for hidden layer
        d_a1 = d_output * w5 * sigmoid_derivative(a1)
        d_a2 = d_output * w6 * sigmoid_derivative(a2)

        dw1 = d_a1 * x1
        dw2 = d_a1 * x2
        db1 = d_a1

        dw3 = d_a2 * x1
        dw4 = d_a2 * x2
        db2 = d_a2

        # Update weights and biases
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
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Predictions
print("\nPredictions:")
for i in range(len(X)):
    x1, x2 = X[i]
    z1 = x1 * w1 + x2 * w2 + b1
    a1 = sigmoid(z1)

    z2 = x1 * w3 + x2 * w4 + b2
    a2 = sigmoid(z2)

    z3 = a1 * w5 + a2 * w6 + b3
    output = sigmoid(z3)

    prediction = 1 if output >= 0.5 else 0
    print(f"Input: {X[i]}, Predicted: {prediction}, Prob: {output:.4f}")
