# BMI	Exercise Level	Heart Disease Risk
# 0.85	  0.15	            1
# 0.65	  0.20	            1
# 0.45	  0.80	            0
# 0.30	  0.90	            0
# 0.70	  0.50	            1
# 0.40	  0.60	            0
# 0.90	  0.10	            1
# 0.35	  0.85	            0
# 0.75	  0.30	            1
# 0.50	  0.70	            0

import numpy as np

X = np.array([
    [0.85, 0.15],
    [0.65, 0.20],
    [0.45, 0.80],
    [0.30, 0.90],
    [0.70, 0.50],
    [0.40, 0.60],
    [0.90, 0.10],
    [0.35, 0.85],
    [0.75, 0.30],
    [0.50, 0.70],
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
    [0],
])

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


# --- Architecture parameters ---
input_size = X.shape[1]    # automatic from data
hidden_size = 2
output_size = 1

# --- Count weights and biases ---
architecture = [input_size, hidden_size, output_size]
total_weights, total_biases = count_weights_and_biases(architecture)
print(f"Total Weights: {total_weights}, Total Biases: {total_biases}")

# --- Initialize weights and biases ---
np.random.seed(42)
W1 = np.random.rand(input_size, hidden_size)    # shape: (input_size, hidden_size)
b1 = np.random.rand(hidden_size)                 # shape: (hidden_size,)
W2 = np.random.rand(hidden_size, output_size)   # shape: (hidden_size, output_size)
b2 = np.random.rand(output_size)                 # shape: (output_size,)

# --- Hyperparameters ---
lr = 0.1
epochs = 10000

# --- Training loop ---
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X)):
        x = X[i]
        target = Y[i][0]  # scalar

        # Forward pass
        z1 = np.dot(x, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        output = sigmoid(z2)[0]  # scalar

        # Compute loss
        loss = binary_cross_entropy(target, output)
        total_loss += loss

        # Backpropagation
        d_output = (output - target) * sigmoid_derivative(output)  # scalar
        dW2 = np.outer(a1, np.array([d_output]))      # (hidden_size, output_size)
        db2 = np.array([d_output])                     # (output_size,)

        d_hidden = np.dot(W2, np.array([d_output])) * sigmoid_derivative(a1)  # (hidden_size,)
        dW1 = np.outer(x, d_hidden)                 # (input_size, hidden_size)
        db1 = d_hidden                              # (hidden_size,)

        # Update weights and biases
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# --- Predictions ---
print("\nPredictions:")
for i in range(len(X)):
    x = X[i]
    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    output = sigmoid(z2)[0]
    prediction = 1 if output >= 0.5 else 0
    print(f"Input: {x}, Predicted: {prediction}, Prob: {output:.4f}")

