"""
Train the final dense neural network on the Amharic base-34 character
dataset, with data augmentation and early stopping. Saves the trained
model and training history for later evaluation/visualization.

Usage:
    python src/train.py
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import os

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_model():
    return keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.RandomRotation(0.05),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(33, activation="softmax"),
    ])


def main():
    X_train = np.load("data/X_train.npy").reshape(-1, 28, 28, 1)
    y_train = np.load("data/y_train.npy")
    X_val = np.load("data/X_val.npy").reshape(-1, 28, 28, 1)
    y_val = np.load("data/y_val.npy")
    X_test = np.load("data/X_test.npy").reshape(-1, 28, 28, 1)
    y_test = np.load("data/y_test.npy")

    model = build_model()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal test accuracy: {test_acc:.4f}")
    print(f"Final test loss: {test_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    model.save("models/amharic_char_model.keras")

    os.makedirs("results", exist_ok=True)
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_dict["test_accuracy"] = float(test_acc)
    history_dict["test_loss"] = float(test_loss)
    with open("results/training_history.json", "w") as f:
        json.dump(history_dict, f, indent=2)

    print("Saved model to models/amharic_char_model.keras")
    print("Saved training history to results/training_history.json")


if __name__ == "__main__":
    main()
