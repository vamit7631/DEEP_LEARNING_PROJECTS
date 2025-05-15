import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

df = pd.read_csv("../../DATASETS/business_sector_dataset.csv")


X = df.iloc[:, : - 1].values
Y = df.iloc[:, - 1].values

sc = StandardScaler()
X = sc.fit_transform(X)

print(X.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation="sigmoid", input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
model.fit(X,Y, epochs = 1000, verbose = 0)

predictions = model.predict(X)

correct = 0
for i, (x, pred) in enumerate(zip(X, predictions)):
    label = 1 if pred > 0.5 else 0
    actual = Y[i]
    if label == actual:
        correct += 1
    print(f"Input {x}, Prediction {label}, Prob {pred[0]:.4f}, Actual: {actual}")

print(f"\nManually Computed Accuracy: {correct / len(Y) * 100:.2f}%")
# final accuracy : 81.20 %