import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import regularizers

# Load data
df = pd.read_csv("../../DATASETS/business_sector_dataset.csv")
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Standardize features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)  # <-- FIX: transform, not fit_transform!

# Build model with Dropout + L2 regularization
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation="relu", kernel_regularizer=regularizers.l2(0.001), input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(4, activation="relu", kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile with lower learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])

# Add EarlyStopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train model
history = model.fit(X_train, Y_train, epochs=500, verbose=0, 
                    validation_data=(X_test, Y_test), callbacks=[early_stop])

# Evaluate
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print("Test Accuracy:", acc)

# Predictions
predictions = model.predict(X_test)
correct = 0
for i, (x, pred) in enumerate(zip(X_test, predictions)):
    label = 1 if pred > 0.5 else 0
    actual = Y_test[i]
    if label == actual:
        correct += 1
    print(f"Input {x}, Prediction {label}, Prob {pred[0]:.4f}, Actual: {actual}")

print(f"\nManually Computed Accuracy: {correct / len(Y_test) * 100:.2f}%")

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
