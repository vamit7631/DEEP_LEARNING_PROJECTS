# # GPA  | Experience | Salary
# 0.91  | 0.87        | 0.90
# 0.78  | 0.72        | 0.76
# 0.66  | 0.61        | 0.65
# 0.58  | 0.45        | 0.52
# 0.88  | 0.92        | 0.90
# 0.73  | 0.81        | 0.77
# 0.65  | 0.59        | 0.63
# 0.52  | 0.49        | 0.51
# 0.93  | 0.88        | 0.92
# 0.85  | 0.83        | 0.85
# 0.48  | 0.42        | 0.44
# 0.35  | 0.30        | 0.33
# 0.72  | 0.68        | 0.70
# 0.63  | 0.66        | 0.64
# 0.57  | 0.51        | 0.54
# 0.81  | 0.76        | 0.79
# 0.47  | 0.39        | 0.43
# 0.32  | 0.28        | 0.30
# 0.68  | 0.74        | 0.71
# 0.55  | 0.58        | 0.56
# 0.82  | 0.79        | 0.80
# 0.49  | 0.44        | 0.47
# 0.36  | 0.35        | 0.35
# 0.67  | 0.63        | 0.65
# 0.77  | 0.81        | 0.79
# 0.59  | 0.50        | 0.54
# 0.92  | 0.95        | 0.93
# 0.86  | 0.82        | 0.84
# 0.74  | 0.66        | 0.70
# 0.61  | 0.55        | 0.58
# 0.44  | 0.40        | 0.42
# 0.39  | 0.31        | 0.35
# 0.70  | 0.73        | 0.71
# 0.53  | 0.48        | 0.50
# 0.30  | 0.25        | 0.28
# 0.79  | 0.75        | 0.77
# 0.66  | 0.62        | 0.64
# 0.58  | 0.49        | 0.54
# 0.84  | 0.87        | 0.86
# 0.50  | 0.41        | 0.46
# 0.42  | 0.37        | 0.40
# 0.60  | 0.53        | 0.57
# 0.76  | 0.79        | 0.78
# 0.69  | 0.65        | 0.67
# 0.62  | 0.60        | 0.61
# 0.46  | 0.39        | 0.43
# 0.38  | 0.33        | 0.35
# 0.80  | 0.77        | 0.78
# 0.71  | 0.69        | 0.70


import numpy as np
from sklearn.model_selection import train_test_split

# Input features: [GPA, Experience]
X = np.array([[0.91,0.87],[0.78,0.72],[0.66,0.61],[0.58,0.45],[0.88,0.92],[0.73,0.81],[0.65,0.59],[0.52,0.49],[0.93,0.88],[0.85,0.83],[0.48,0.42],[0.35,0.30],[0.72,0.68],[0.63,0.66],[0.57,0.51],[0.81,0.76],[0.47,0.39],[0.32,0.28],[0.68,0.74],[0.55,0.58],[0.82,0.79],[0.49,0.44],[0.36,0.35],[0.67,0.63],[0.77,0.81],[0.59,0.50],[0.92,0.95],[0.86,0.82],[0.74,0.66],[0.61,0.55],[0.44,0.40],[0.39,0.31],[0.70,0.73],[0.53,0.48],[0.30,0.25],[0.79,0.75],[0.66,0.62],[0.58,0.49],[0.84,0.87],[0.50,0.41],[0.42,0.37],[0.60,0.53],[0.76,0.79],[0.69,0.65],[0.62,0.60],[0.46,0.39],[0.38,0.33],[0.80,0.77],[0.71,0.69]])

Y = np.array([[0.90],[0.76],[0.65],[0.52],[0.90],[0.77],[0.63],[0.51],[0.92],[0.85],[0.44],[0.33],[0.70],[0.64],[0.54],[0.79],[0.43],[0.30],[0.71],[0.56],[0.80],[0.47],[0.35],[0.65],[0.79],[0.54],[0.93],[0.84],[0.70],[0.58],[0.42],[0.35],[0.71],[0.50],[0.28],[0.77],[0.64],[0.54],[0.86],[0.46],[0.40],[0.57],[0.78],[0.67],[0.61],[0.43],[0.35],[0.78],[0.70]])

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

lr = 0.1
epochs = 10000

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):  # assuming x is sigmoid(x)
    return x * (1 - x)

for epoch in range(epochs):
    total_train_loss = 0
    total_test_loss = 0
    for i in range(len(X_train)):
        x1, x2 = X_train[i]
        target = Y_train[i][0]

        # Forward pass
        z1 = x1 * w1 + x2 * w2 + b1
        a1 = sigmoid(z1)

        z2 = x1 * w3 + x2 * w4 + b2
        a2 = sigmoid(z2)

        z3 = a1 * w5 + a2 * w6 + b3
        output = sigmoid(z3)

        # Compute loss
        train_loss = (target - output) ** 2   # mean squared error loss function
        total_train_loss += train_loss

        # Backpropagation
        d_output = 2 * (output - target) * sigmoid_derivative(output)

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

    for i in range(len(X_test)):
        x1, x2 = X_test[i]
        target = Y_test[i][0]

        # Forward pass
        z1 = x1 * w1 + x2 * w2 + b1
        a1 = sigmoid(z1)

        z2 = x1 * w3 + x2 * w4 + b2
        a2 = sigmoid(z2)

        z3 = a1 * w5 + a2 * w6 + b3
        output = sigmoid(z3)

        test_loss = (target - output) ** 2
        total_test_loss += test_loss
    
    avg_train_loss = total_train_loss / len(X_train)
    avg_test_loss = total_test_loss / len(X_test)

    if epoch % 1000 == 0:
        print(f"Epochs {epoch} , Total Train Loss {avg_train_loss:.4f} , Total Test Loss {avg_test_loss :.4f}")

print("\nPredictions:")
for i in range(len(X_test)):
    x1, x2 = X_test[i]

    z1 = w1 * x1 + w2 * x2 + b1
    a1 = sigmoid(z1)

    z2 = x1 * w3 + x2 * w4 + b2
    a2 = sigmoid(z2)

    z3 = a1 * w5 + a2 * w6 + b3
    output = sigmoid(z3)


    print(f"Input: {X_test[i]}, Predicted Salary: {output * 100000:.2f} ")