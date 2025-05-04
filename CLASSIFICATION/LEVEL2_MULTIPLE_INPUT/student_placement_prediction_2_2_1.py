# GPA | Experience | Placement (normalized)
# 0.9 | 0.8        | 1
# 0.6 | 0.5        | 1
# 0.4 | 0.2        | 0
# 0.7 | 0.9        | 0


import numpy as np

# --- Helper: Count weights and biases ---
def count_weights_and_biases(lers):
    total_weights = 0
    total_biases = 0
    for i in range(1, len(lers)):
        total_weights += lers[i - 1] * lers[i]
        total_biases += lers[i]
    return total_weights, total_biases

# --- Activation functions ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):  # assuming x is sigmoid(x)
    return x * (1 - x)

def binary_cross_entropy(y_true, y_pred):
    return -(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))


# --- Dataset (with 4 input features) ---
X = np.array([
    [0.9, 0.8, 0.7, 0.6],
    [0.6, 0.5, 0.4, 0.3],
    [0.4, 0.2, 0.1, 0.2],
    [0.7, 0.9, 0.8, 0.7]
])

y = np.array([
    [1],
    [1],
    [0],
    [0]
])

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
        target = y[i][0]  # scalar

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
