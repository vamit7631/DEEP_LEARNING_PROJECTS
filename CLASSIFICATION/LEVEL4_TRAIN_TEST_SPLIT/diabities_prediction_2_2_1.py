import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("../../DATASETS/diabetes_prediction_dataset.csv")

def standardScaler(x):
    mean = x.mean()
    std = x.std()
    return (x - mean) / std

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def binary_cross_entropy(y_true, y_pred):
    return -(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))

df["bmi"] = standardScaler(df["bmi"])
df["HbA1c_level"] = standardScaler(df["HbA1c_level"])


X = df[["bmi","HbA1c_level"]].values
Y = df["diabetes"].values


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

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

lr = 0.01
epochs = 10000

train_losses = []
test_losses = []
val_accuracies = []

for epoch in range(epochs):
    total_train_loss = 0
    total_test_loss = 0
    correct_val = 0
    for i in range(len(X_train)):
        x1, x2 = X_train[i]
        target = Y_train[i]
        
        z1 = w1 * x1 + w2 * x2 + b1
        a1 = sigmoid(z1)

        z2 = w3 * x1 + w4 * x2 + b2
        a2 = sigmoid(z2)

        z3 = w5 * a1 + w6 * a2 + b3
        output = sigmoid(z3)


        loss = binary_cross_entropy(target, output)
        total_train_loss += loss

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

        w1 -= lr * dw1
        w2 -= lr * dw2
        w3 -= lr * dw3
        w4 -= lr * dw4
        w5 -= lr * dw5
        w6 -= lr * dw6
        b1 -= lr * db1
        b2 -= lr * db2
        b3 -= lr * db3


    for i in range(len(X_test)):
        x1, x2 = X_test[i]
        target = Y_test[i]

        z1 = w1 * x1 + w2 * x2 + b1
        a1 = sigmoid(z1)

        z2 = w3 * x1 + w4 * x2 + b2
        a2 = sigmoid(z2)

        z3 = w5 * a1 + w6 * a2 + b3
        output = sigmoid(z3)


        loss = binary_cross_entropy(target, output)
        total_test_loss += loss


        prediction = 1 if output > 0.5 else 0
        if prediction == target:
            correct_val += 1

    avg_train_loss = total_train_loss / len(X_train)
    avg_test_loss = total_test_loss / len(X_test)
    val_accuracy = correct_val / len(X_test)

    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    val_accuracies.append(val_accuracy)

    if epoch % 1000 == 0:
        print(f"Epochs {epoch}, Train Loss: {avg_train_loss:.4f}, Test Loss:  {avg_test_loss:.4f}, Val Acc: {val_accuracy:.4f}")

print("\nFinal Predictions on Validation Set:")
correct = 0

for i in range(len(X_test)):
    x1, x2 = X_test[i]
    
    z1 = w1 * x1 + w2 * x2 + b1
    a1 = sigmoid(z1)

    z2 = w3 * x1 + w4 * x2 + b2
    a2 = sigmoid(z2)

    z3 = w5 * a1 + w6 * a2 + b3
    output = sigmoid(z3)

    prediction = 1 if output > 0.5 else 0
    if prediction == Y_test[i]:
        correct += 1

    print(f"Input: [{x1:.2f}, {x2:.2f}], Predicted: {prediction}, Prob: {output:.4f}")

accuracy = correct / len(X_test)
print(f"\nFinal Validation Accuracy: {accuracy * 100:.2f}%")
# final accuracy : 95.50 %

# Plotting loss and accuracy curves
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Validation Loss')
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

