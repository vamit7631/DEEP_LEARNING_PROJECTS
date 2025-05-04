# GPA | Experience | Extra | Placement
# 0.9 | 0.8        | 0.5   | 1
# 0.6 | 0.5        | 0.3   | 1
# 0.4 | 0.2        | 0.1   | 0
# 0.7 | 0.9        | 0.6   | 0

import numpy as np 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
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
Y = np.array([
    [1],
    [1],
    [0],
    [0]
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
w9 = np.random.rand()
w10 = np.random.rand()
w11 = np.random.rand()
w12 = np.random.rand()
b1 = np.random.rand()
b2 = np.random.rand()
b3 = np.random.rand()
b4 = np.random.rand()


lr = 0.1
epochs = 10000

for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X)):
        x1, x2, x3 = X[i]
        target = Y[i][0]

        z1 = w1 * x1 + w2 * x2 + w3 * x3 + b1
        a1 = sigmoid(z1)

        z2 = w4 * x1 + w5 * x2 + w6 * x3 + b2
        a2 = sigmoid(z2)

        z3 = w7 * x1 + w8 * x2 + w9 * x3 + b3
        a3 = sigmoid(z3)

        z4 = a1 * w10 + a2 * w11 + a3 * w12 + b4
        output = sigmoid(z4)

        loss = binary_cross_entropy(target, output)
        total_loss += loss

        d_output = (output - target) * sigmoid_derivative(output)

        dw10 = d_output * a1
        dw11 = d_output * a2
        dw12 = d_output * a3
        db4 = d_output

        d_a1 = d_output * w10 * sigmoid_derivative(a1)
        d_a2 = d_output * w11 * sigmoid_derivative(a2)
        d_a3 = d_output * w12 * sigmoid_derivative(a3)

        dw1 = d_a1 * x1
        dw2 = d_a1 * x2
        dw3 = d_a1 * x3
        db1 = d_a1

        dw4 = d_a2 * x1
        dw5 = d_a2 * x2
        dw6 = d_a2 * x3
        db2 = d_a2

        dw7 = d_a3 * x1
        dw8 = d_a3 * x2
        dw9 = d_a3 * x3
        db3 = d_a3

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
        w9 -= lr * dw9
        b3 -= lr * db3

        w10 -= lr * dw10
        w11 -= lr * dw11
        w12 -= lr * dw12
        b4 -= lr * db4

    if epoch % 1000 == 0:
        print(f"Epoch : { epoch }, Loss : {total_loss:.4f} ")

    
print("\nPrediction")

for i in range(len(X)):
    x1, x2, x3 = X[i]
        
    z1 = w1 * x1 + w2 * x2 + w3 * x3 + b1
    a1 = sigmoid(z1)

    z2 = w4 * x1 + w5 * x2 + w6 * x3 + b2
    a2 = sigmoid(z2)

    z3 = w7 * x1 + w8 * x2 + w9 * x3 + b3
    a3 = sigmoid(z3)

    z4 = a1 * w10 + a2 * w11 + a3 * w12 + b4
    output = sigmoid(z4)

    prediction = 1 if output >= 0.5 else 0
    print(f"Input {X[i]}, predicted value {prediction} , output {output:.4f}")
