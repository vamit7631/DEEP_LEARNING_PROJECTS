import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load dataset
df = pd.read_csv("../../DATASETS/business_sector_dataset.csv")

# Prepare features and labels
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Standardize
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)  # use transform, not fit_transform

# Define hyperparameter grid
neurons = [8, 16]
dropouts = [0.2, 0.3]
l2_regs = [0.001, 0.0001]
learning_rates = [0.001, 0.0005]

# Track best
best_acc = 0
best_params = {}

# Initialize history records list
history_records = []

for n, d, l2r, lr in itertools.product(neurons, dropouts, l2_regs, learning_rates):
    print(f"Testing config: neurons={n}, dropout={d}, l2={l2r}, lr={lr}")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(n, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2r), input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(d),
        tf.keras.layers.Dense(n//2, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2r)),
        tf.keras.layers.Dropout(d),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    
    history = model.fit(
        X_train, Y_train,
        epochs=200,
        validation_data=(X_test, Y_test),
        callbacks=[early_stop],
        verbose=0
    )
    
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    
    # Record history with label
    history_records.append((f"neurons={n},dropout={d},l2={l2r},lr={lr}", history))

    if acc > best_acc:
        best_acc = acc
        best_params = {'neurons': n, 'dropout': d, 'l2': l2r, 'lr': lr}
        print(f"🔥 New best acc: {acc:.4f} with params {best_params}")

print("\n✅ Best overall configuration:")
print(f"Params: {best_params}")
print(f"Test Accuracy: {best_acc:.4f}")


# Plot accuracy and loss per run
plt.figure(figsize=(14, 6))

for label, hist in history_records:
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['accuracy'], label=f'{label}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy per Run')
    
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'], label=f'{label}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss per Run')

plt.subplot(1, 2, 1)
plt.legend(fontsize='small', loc='lower right')

plt.subplot(1, 2, 2)
plt.legend(fontsize='small', loc='upper right')

plt.tight_layout()
plt.show()

print("\n✅ Best overall configuration:")
print(f"Params: {best_params}")
print(f"Test Accuracy: {best_acc:.4f}")