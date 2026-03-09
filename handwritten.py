import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import matplotlib.pyplot as plt

#! 1. Initiliazation
# TODO: Load the mnist dataset
(X_t, y_t), (X_r, y_r) = tf.keras.datasets.mnist.load_data()
X_all = np.concatenate((X_t, X_r), axis=0)
y_all = np.concatenate((y_t, y_r), axis=0)

# TODO: Split the data into train, dev and test sets
from sklearn.model_selection import train_test_split as tts

X_train, X_rem, y_train, y_rem = tts(X_all, y_all, train_size=0.75, random_state=42)
X_dev, X_test, y_dev, y_test = tts(X_rem, y_rem, train_size=0.5, random_state=42)

# TODO: Shuffle and normalize the data
X_train = X_train / 255
X_dev = X_dev / 255
X_test = X_test / 255

# * Copy for Conv2D
X_train_c = X_train.reshape(-1, 28, 28, 1)
X_dev_c = X_dev.reshape(-1, 28, 28, 1)
X_test_c = X_test.reshape(-1, 28, 28, 1)

# TODO: Reshape
X_train = X_train.reshape(-1, 784)
X_dev = X_dev.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)

#! 2. Train the model
# TODO: Layers!
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

model = Sequential(
    [
        Dense(units=128, activation="relu", input_shape=(784,)),
        Dense(units=64, activation="relu"),
        Dense(units=10, activation="softmax"),
    ]
)

# TODO: Complie
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy as scc

model.compile(optimizer=Adam(1e-3), loss=scc(), metrics=["accuracy"])

# TODO: Callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callback = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint("best_no_reg.keras"),
]

# TODO: Fit!
history = model.fit(
    X_train,
    y_train,
    epochs=6,
    batch_size=32,
    validation_data=(X_dev, y_dev),
    callbacks=callback,
)

# TODO: Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)

#!3. Visualize
history_dict = history.history
plt.figure(figsize=(8, 5))
plt.plot(history_dict["accuracy"], label="Train Accuracy", linewidth=2)
plt.plot(history_dict["val_accuracy"], label="Validation Accuracy", linewidth=2)
# Add test accuracy as horizontal line
plt.axhline(y=test_acc, color="red", linestyle="--", label="Test Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
model.save("model_no_reg.keras")

#!Part II: Adding Regularization to improve the model

# TODO: Adding Regulization Term
model_reg = Sequential(
    [
        Dense(units=128, activation="relu", input_shape=(784,)),
        Dropout(0.2),
        Dense(units=64, activation="relu"),
        Dropout(0.2),
        Dense(units=10, activation="softmax"),
    ]
)

# TODO: Compile the model
model_reg.compile(optimizer=Adam(1e-3), loss=scc(), metrics=["accuracy"])
callback_reg = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint("best_reg.keras"),
]

# TODO: Train the model
history_reg = model_reg.fit(
    X_train,
    y_train,
    epochs=6,
    validation_data=(X_dev, y_dev),
    batch_size=32,
    callbacks=callback_reg,
)

# TODO: Evaluate
test_loss_reg, test_acc_reg = model_reg.evaluate(X_test, y_test)

# Plotting
history_for_plotting = history_reg.history
plt.figure(figsize=(12, 5))

# ---- Accuracy Plot ----
plt.subplot(1, 2, 1)

plt.plot(history_dict["accuracy"], label="Train (No Reg)", linewidth=2)
plt.plot(history_dict["val_accuracy"], label="Val (No Reg)", linewidth=2)

plt.plot(
    history_for_plotting["accuracy"], linestyle="--", label="Train (Reg)", linewidth=2
)
plt.plot(
    history_for_plotting["val_accuracy"], linestyle="--", label="Val (Reg)", linewidth=2
)

plt.axhline(y=test_acc, color="red", linestyle=":", label="Test (No Reg)")
plt.axhline(y=test_acc_reg, color="green", linestyle=":", label="Test (Reg)")

plt.title("Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

# ---- Loss Plot ----
plt.subplot(1, 2, 2)

plt.plot(history_dict["loss"], label="Train Loss (No Reg)", linewidth=2)
plt.plot(history_dict["val_loss"], label="Val Loss (No Reg)", linewidth=2)

plt.plot(
    history_for_plotting["loss"], linestyle="--", label="Train Loss (Reg)", linewidth=2
)
plt.plot(
    history_for_plotting["val_loss"],
    linestyle="--",
    label="Val Loss (Reg)",
    linewidth=2,
)

plt.title("Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#! PART III: Using Convolution Layer
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten

model_cnn = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax"),
    ]
)
callback_cnn = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint("bestcnn_reg.keras"),
]
model_cnn.compile(optimizer=Adam(1e-3), loss=scc(), metrics=["accuracy"])
history_cnn = model_cnn.fit(
    X_train_c,
    y_train,
    validation_data=(X_dev_c, y_dev),
    epochs=8,
    batch_size=32,
    callbacks=callback_cnn,
)
test_loss_cnn, test_acc_cnn = model_cnn.evaluate(X_test_c, y_test)
model_cnn.save("best_cnn.keras")

history_cnn_dict = history_cnn.history
plt.figure(figsize=(14, 6))

# -------- ACCURACY --------
plt.subplot(1, 2, 1)

# MLP
plt.plot(history_dict["val_accuracy"], label="MLP Val", linewidth=2)
plt.plot(history_for_plotting["val_accuracy"], label="MLP + Dropout Val", linewidth=2)
plt.plot(history_cnn_dict["val_accuracy"], label="CNN Val", linewidth=2)

# Test lines
plt.axhline(test_acc, linestyle=":", color="blue", label="MLP Test")
plt.axhline(test_acc_reg, linestyle=":", color="orange", label="Dropout Test")
plt.axhline(test_acc_cnn, linestyle=":", color="green", label="CNN Test")

plt.title("Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

# -------- LOSS --------
plt.subplot(1, 2, 2)

plt.plot(history_dict["val_loss"], label="MLP Val Loss", linewidth=2)
plt.plot(history_for_plotting["val_loss"], label="MLP + Dropout Val Loss", linewidth=2)
plt.plot(history_cnn_dict["val_loss"], label="CNN Val Loss", linewidth=2)

plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

y_pred_probs_cnn = model_cnn.predict(X_test_c)
y_pred_cnn = np.argmax(y_pred_probs_cnn, axis=1)

mis_idx = np.where(y_pred_cnn != y_test)[0]

plt.figure(figsize=(10, 5))
for i in range(10):
    idx = mis_idx[i]
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap="gray")
    plt.title(f"T:{y_test[idx]} P:{y_pred_cnn[idx]}")
    plt.axis("off")

plt.tight_layout()
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns

# Predictions
y_pred_probs = model_cnn.predict(X_test_c)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (CNN)")
plt.show()

# For Recall, Precision and F1-score
from sklearn.metrics import classification_report

report = classification_report(y_test, y_pred)
print(report)
