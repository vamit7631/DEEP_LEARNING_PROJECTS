# GPA | Experience | Extra | Placement
# 0.9 | 0.8        | 0.5   | 1
# 0.6 | 0.5        | 0.3   | 1
# 0.4 | 0.2        | 0.1   | 0
# 0.7 | 0.9        | 0.6   | 0


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

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):  # assuming x is sigmoid(x)
    return x * (1 - x)

def binary_cross_entropy(y_true, y_pred):
    return -(y_true * np.log(y_pred + 1e-9) +  (1 - y_true) * np.log(1 - y_pred + 1e-9))


# Input features: [GPA, Experience]
X = np.array([
    [0.9, 0.8, 0.5],
    [0.6, 0.5, 0.3],
    [0.4, 0.2, 0.1],
    [0.7, 0.9, 0.6]
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
w5 = np.random.rand()
w6 = np.random.rand()
w7 = np.random.rand()
w8 = np.random.rand()
b1 = np.random.rand()
b2 = np.random.rand()
b3 = np.random.rand()

lr = 0.1
epochs = 10000

# Training
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X)):
        x1, x2, x3 = X[i]
        target = y[i][0]

        # Forward pass
        z1 = x1 * w1 + x2 * w2 + x3 * w3 + b1
        a1 = sigmoid(z1)

        z2 = x1 * w4 + x2 * w5 + x3 * w6 + b2
        a2 = sigmoid(z2)

        z3 = a1 * w7 + a2 * w8 + b3
        output = sigmoid(z3)

        # Compute loss
        loss = binary_cross_entropy(target, output)   # binary cross entropy loss function
        total_loss += loss

        # Backpropagation
        d_output = (output - target) * sigmoid_derivative(output)

        # Gradients for output layer
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

    prediction = 1 if output >= 0.5 else 0
    print(f"Input: {X[i]}, Predicted: {prediction}, Prob: {output:.4f}")
