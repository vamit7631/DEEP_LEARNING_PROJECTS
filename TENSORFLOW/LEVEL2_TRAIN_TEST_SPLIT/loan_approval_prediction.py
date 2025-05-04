import numpy as np 
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('../../DATASETS/loan_approval_dataset.csv')

le = LabelEncoder()
df['loan_status']= le.fit_transform(df['loan_status'])

X = df[["income_annum","loan_amount", "luxury_assets_value", "bank_asset_value"]].values
Y = df[["loan_status"]].astype(np.float32).values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(4,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam" , loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, Y_train, epochs= 1000, verbose = 0)

loss, acc = model.evaluate(X_test, Y_test, verbose = 0)
print(f"Test accuracy : {acc}")

predictions = model.predict(X_test)


for x, pred in enumerate(zip(X_test, predictions)):
    label = 1 if pred[0] > 0.5 else 0
    print(f"Input {x}, Prediction {label}, Prob {pred[0]:.4f}")
