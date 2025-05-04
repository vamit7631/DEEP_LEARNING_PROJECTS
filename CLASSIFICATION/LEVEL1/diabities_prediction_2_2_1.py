import numpy as np 
import pandas as pd


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
Y = df[["diabetes"]].values

lr = 0.1
epochs = 10000

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




for epoch in range(epochs):
    total_loss = 0    
    for i in range(len(X)):
        x1, x2 = X[i]
        target = Y[i][0]


        z1 = w1 * x1 + w2 * x2 + b1
        a1 = sigmoid(z1)

        z2 = w3 * x1 + w4 * x2 + b2
        a2 = sigmoid(z2)

        z3 = w5 * a1 + w6 * a2 + b3
        output = sigmoid(z3)

        loss = binary_cross_entropy(target, output)
        total_loss += loss

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

    avg_loss = total_loss/len(X)
    if epoch % 1000 == 0:
        print(f"Epochs {epoch} , loss {avg_loss:.4f}")

print("\nPrediction")
correct = 0
for i in range(len(X)):
    x1, x2 = X[i]

    z1 = w1 * x1 + w2 * x2 + b1
    a1 = sigmoid(z1)

    z2 = w3 * x1 + w4 * x2 + b2
    a2 = sigmoid(z2)

    z3 = w5 * a1 + w6 * a2 + b3
    output = sigmoid(z3)

    prediction = 1 if output > 0.5 else 0
    if prediction == Y[i][0]:
        correct += 1 
    # print(f"Input : {X[i]} Prediction {prediction} Prob {output:.4f}")


accuracy = correct / len(X)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
# final accuracy : 95.47 %