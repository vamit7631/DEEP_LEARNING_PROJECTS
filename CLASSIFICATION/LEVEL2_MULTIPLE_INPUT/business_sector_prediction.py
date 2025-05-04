import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../../DATASETS/business_sector_dataset.csv")

def standardScaler(x):
    mean = x.mean()
    std = x.std()
    return (x - mean) / std

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def binary_cross_entropy(y_true, y_pred):
    return -(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))


X = df.iloc[:, : - 1 ].values
Y = df.iloc[:, -1].values

X = standardScaler(X)
print(X.shape)

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
        target = Y[i]

        z1 = np.dot(x, w1) + b1
        a1 = sigmoid(z1)

        z2 = np.dot(a1, w2) + b2
        output = sigmoid(z2)[0]

        loss = binary_cross_entropy(target, output)
        total_loss += loss

        d_output = (output - target) * sigmoid_derivative(output)

        dw2 = np.outer(a1, np.array([d_output]))
        db2 = np.array([d_output])

        d_hidden = np.dot(w2, np.array([d_output])) * sigmoid_derivative(a1)
        dw1 = np.outer(x, d_hidden)
        db1 = d_hidden

        w1 -= lr * dw1
        w2 -= lr * dw2
        b1 -= lr * db1
        b2 -= lr * db2

    avg_loss = total_loss/len(X)
    if epoch % 1000 == 0:
        print(f"Total Epochs {epoch} , Total Loss: {avg_loss:.4f}")


print("\nPredictions")
correct = 0
for i in range(len(X)):
    x = X[i]

    z1 = np.dot(x, w1) + b1
    a1 = sigmoid(z1)

    z2 = np.dot(a1, w2) + b2
    output = sigmoid(z2)[0]

    prediction = 1 if output > 0.5 else 0
    if prediction == Y[i]:
        correct += 1 
    print(f"Input {x}, predicted {prediction} , Prob {output:.4f}")


accuracy = correct / len(X)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
# final accuracy : 81.20 %
