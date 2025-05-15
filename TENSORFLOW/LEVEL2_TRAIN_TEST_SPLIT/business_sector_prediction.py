import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

df = pd.read_csv("../../DATASETS/business_sector_dataset.csv")


X = df.iloc[:, : - 1].values
Y = df.iloc[:, - 1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

print(X.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation="relu", input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
history = model.fit(X_train,Y_train, epochs = 1000, verbose = 0, validation_data=(X_test, Y_test))

loss, acc = model.evaluate(X_test, Y_test, verbose = 0)
print(f"Test Accuracy: {acc * 100:.2f}%")

predictions = model.predict(X_test)


correct = 0
for i, (x, pred) in enumerate(zip(X_test, predictions)):
    label = 1 if pred > 0.5 else 0
    actual = Y_test[i]
    if label == actual:
        correct += 1
    print(f"Input {x}, Prediction {label}, Prob {pred[0]:.4f}, Actual: {actual}")

print(f"\nManually Computed Accuracy: {correct / len(Y_test) * 100:.2f}%")
# final accuracy : 81.50 %

plt.figure(figsize=(14, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()

plt.tight_layout()
plt.show()