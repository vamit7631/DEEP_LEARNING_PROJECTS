import numpy as np

# --- Activation functions ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def binary_cross_entropy(y_true, y_pred):
    return -(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9)).mean()


# --- MLP Class ---
class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        np.random.seed(42)
        self.W1 = np.random.rand(input_size, hidden_size)
        self.b1 = np.random.rand(1, hidden_size)
        self.W2 = np.random.rand(hidden_size, output_size)
        self.b2 = np.random.rand(1, output_size)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = sigmoid(self.z2)
        return self.output

    def backward(self, X, y, lr):
        m = X.shape[0]
        d_output = (self.output - y) * sigmoid_derivative(self.output)
        dW2 = np.dot(self.a1.T, d_output) / m
        db2 = np.sum(d_output, axis=0, keepdims=True) / m

        d_hidden = np.dot(d_output, self.W2.T) * sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, d_hidden) / m
        db1 = np.sum(d_hidden, axis=0, keepdims=True) / m

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X, y, epochs, lr):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = binary_cross_entropy(y, y_pred)
            self.backward(X, y, lr)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        output = self.forward(X)
        return (output >= 0.5).astype(int), output


# --- Dataset ---
X = np.array([
    [0.9, 0.8],
    [0.6, 0.5],
    [0.4, 0.2],
    [0.7, 0.9]
])

y = np.array([
    [1],
    [1],
    [0],
    [0]
])

# --- Initialize and train model ---
model = SimpleMLP(input_size=2, hidden_size=2, output_size=1)
model.train(X, y, epochs=10000, lr=0.1)

# --- Predictions ---
print("\nPredictions:")
pred_labels, pred_probs = model.predict(X)
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted: {pred_labels[i][0]}, Prob: {pred_probs[i][0]:.4f}")
