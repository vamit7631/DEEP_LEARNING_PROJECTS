import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

df = pd.read_csv('../../DATASETS/loan_approval_dataset.csv')


le = LabelEncoder()
df['loan_status']= le.fit_transform(df['loan_status'])



X = df[["income_annum","loan_amount", "luxury_assets_value", "bank_asset_value"]].values
Y = df[["loan_status"]].astype(np.float32).values


sc = StandardScaler()
X = sc.fit_transform(X)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(4,)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])


model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, Y, epochs=1000, verbose = 0)

predictions = model.predict(X)

correct = 0
for i, (x, pred) in enumerate(zip(X, predictions)):
    label = 1 if pred > 0.5 else 0
    actual = Y[i][0]
    if label == actual:
        correct += 1
    print(f"Input {x}, Prediction {label}, Prob {pred[0]:.4f}, Actual: {actual}")

print(f"\nManually Computed Accuracy: {correct / len(Y) * 100:.2f}%")