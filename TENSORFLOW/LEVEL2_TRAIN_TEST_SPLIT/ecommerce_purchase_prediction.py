import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

X = np.array([
    [0.90, 0.85, 0.70, 0.60],
    [0.70, 0.60, 0.40, 0.35],
    [0.30, 0.20, 0.10, 0.15],
    [0.40, 0.25, 0.30, 0.20],
    [0.80, 0.75, 0.65, 0.60],
    [0.35, 0.40, 0.30, 0.25],
    [0.95, 0.90, 0.80, 0.85],
    [0.50, 0.45, 0.40, 0.35],
    [0.85, 0.80, 0.75, 0.70],
    [0.60, 0.55, 0.50, 0.45],
], dtype=np.float32)

Y = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 1], dtype=np.float32)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation="sigmoid", input_shape=(4,)),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, Y_train, epochs = 1000, verbose=0)

loss, acc = model.evaluate(X_test, Y_test, verbose = 0)
print("Test Accuracy", acc)

predictions = model.predict(X_test)


for x, pred in zip(X_test, predictions):
    label = 1 if pred[0] > 0.5 else 0
    print(f"Input {x}, Prediction {label}, Prob {pred[0]:.4f}")