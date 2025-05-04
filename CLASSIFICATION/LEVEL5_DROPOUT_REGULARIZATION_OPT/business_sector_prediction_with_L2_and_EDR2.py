import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../../DATASETS/business_sector_dataset.csv")

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def binary_cross_entropy(y_true, y_pred):
    return -(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

input_size = X.shape[1]
hidden_size = 16  # increased hidden size
output_size = 1

np.random.seed(42)
w1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros(hidden_size)
w2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros(output_size)

lr = 0.01
epochs = 10000
l2_lambda = 0.001  # L2 regularization strength
dropout_rate = 0.3

train_losses = []
test_losses = []
val_accuracies = []

for epoch in range(epochs):
    total_train_loss = 0
    total_test_loss = 0
    correct_val = 0

    for i in range(len(X_train)):
        x = X_train[i]
        target = Y_train[i]

        # forward
        z1 = np.dot(x, w1) + b1
        a1 = relu(z1)
        dropout_mask = np.random.binomial(1, 1 - dropout_rate, size=a1.shape)
        a1 *= dropout_mask  # apply dropout

        z2 = np.dot(a1, w2) + b2
        output = sigmoid(z2)[0]

        # loss + L2 penalty
        loss = binary_cross_entropy(target, output)
        l2_penalty = (l2_lambda / 2) * (np.sum(w1 ** 2) + np.sum(w2 ** 2))
        total_train_loss += loss + l2_penalty

        # backward
        d_output = (output - target) * sigmoid_derivative(output)

        dw2 = np.outer(a1, np.array([d_output])) + l2_lambda * w2
        db2 = np.array([d_output])

        d_hidden = np.dot(w2, np.array([d_output])) * relu_derivative(a1)
        dw1 = np.outer(x, d_hidden) + l2_lambda * w1
        db1 = d_hidden

        w1 -= lr * dw1
        w2 -= lr * dw2
        b1 -= lr * db1
        b2 -= lr * db2

    # validation
    for i in range(len(X_test)):
        x = X_test[i]
        target = Y_test[i]

        z1 = np.dot(x, w1) + b1
        a1 = relu(z1)
        z2 = np.dot(a1, w2) + b2
        output = sigmoid(z2)[0]

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
        print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Val Acc: {val_accuracy:.4f}")

print("\nFinal Predictions on Validation Set:")
correct = 0
for i in range(len(X_test)):
    x = X_test[i]

    z1 = np.dot(x, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    output = sigmoid(z2)[0]

    prediction = 1 if output > 0.5 else 0
    if prediction == Y_test[i]:
        correct += 1
    print(f"Input {x}, predicted {prediction}, Prob {output:.4f}")

accuracy = correct / len(X_test)
print(f"\nFinal Validation Accuracy: {accuracy * 100:.2f}%")
# final accuracy : 81.50 (better accuracy without overfitting)

# Plotting
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
