import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load and clean dataset
df = pd.read_csv("../../DATASETS/Social_Network_Ads.csv")

# Standardization function
def standardScaler(x):
    mean = x.mean()
    std = x.std()
    return (x - mean) / std

# Apply scaling to Age and EstimatedSalary
df['Age'] = standardScaler(df['Age'])
df['EstimatedSalary'] = standardScaler(df['EstimatedSalary'])

# Extract inputs (features) and outputs (target)
X = df[['Age', 'EstimatedSalary']].values
Y = df['Purchased'].values

# Split into training and validation sets (80/20 split)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Set random seed
np.random.seed(42)

# Initialize weights and biases
w1, w2 = np.random.rand(), np.random.rand()
w3, w4 = np.random.rand(), np.random.rand()
w5, w6 = np.random.rand(), np.random.rand()
b1, b2, b3 = np.random.rand(), np.random.rand(), np.random.rand()

# Hyperparameters
lr = 0.1
epochs = 10000

# Activation and loss functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def binary_cross_entropy(y_true, y_pred):
    return -(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))

# Lists to store loss and accuracy over epochs
train_losses = []
val_losses = []
val_accuracies = []

# Training loop
for epoch in range(epochs):
    total_train_loss = 0
    total_val_loss = 0
    correct_val = 0
    
    # Training phase
    for i in range(len(X_train)):
        x1, x2 = X_train[i]
        target = Y_train[i]

        # Forward Propagation
        z1 = w1 * x1 + w2 * x2 + b1
        a1 = sigmoid(z1)

        z2 = w3 * x1 + w4 * x2 + b2
        a2 = sigmoid(z2)

        z3 = w5 * a1 + w6 * a2 + b3
        output = sigmoid(z3)

        # Loss
        loss = binary_cross_entropy(target, output)
        total_train_loss += loss

        # BackPropagation
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

    # Validation phase
    for i in range(len(X_val)):
        x1, x2 = X_val[i]
        target = Y_val[i]

        z1 = w1 * x1 + w2 * x2 + b1
        a1 = sigmoid(z1)

        z2 = w3 * x1 + w4 * x2 + b2
        a2 = sigmoid(z2)

        z3 = w5 * a1 + w6 * a2 + b3
        output = sigmoid(z3)

        loss = binary_cross_entropy(target, output)
        total_val_loss += loss

        prediction = 1 if output >= 0.5 else 0
        if prediction == target:
            correct_val += 1

    avg_train_loss = total_train_loss / len(X_train)
    avg_val_loss = total_val_loss / len(X_val)
    val_accuracy = correct_val / len(X_val)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    # Print every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

# Final predictions and accuracy
print("\nFinal Predictions on Validation Set:")
correct = 0
for i in range(len(X_val)):
    x1, x2 = X_val[i]

    z1 = w1 * x1 + w2 * x2 + b1
    a1 = sigmoid(z1)

    z2 = w3 * x1 + w4 * x2 + b2
    a2 = sigmoid(z2)

    z3 = w5 * a1 + w6 * a2 + b3
    output = sigmoid(z3)

    prediction = 1 if output >= 0.5 else 0
    if prediction == Y_val[i]:
        correct += 1
    print(f"Input: [{x1:.2f}, {x2:.2f}], Predicted: {prediction}, Prob: {output:.4f}")

accuracy = correct / len(X_val)
print(f"\nFinal Validation Accuracy: {accuracy * 100:.2f}%")

# Plotting loss and accuracy curves
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()
