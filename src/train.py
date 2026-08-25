"""
Train the neural network (Dense or CNN) on the Amharic base character
dataset, with data augmentation and early stopping. Saves the trained
model and training history for later evaluation/visualization.

Usage:
    python src/train.py                # Train default dense model
    python src/train.py --model dense  # Train dense model
    python src/train.py --model cnn    # Train CNN model
"""

import argparse
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_dense_model():
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


# Backwards-compatible alias
build_model = build_dense_model


def build_cnn_model():
    return keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.RandomRotation(0.05),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomZoom(0.1),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(33, activation="softmax"),
    ])


from sklearn.utils.class_weight import compute_class_weight


def oversample_class_de(X_train, y_train, class_names, target_count=None, seed=SEED):
    idx_de = int(np.where(class_names == "169de")[0][0])
    idx_te = int(np.where(class_names == "211Te")[0][0])
    
    de_indices = np.where(y_train == idx_de)[0]
    te_indices = np.where(y_train == idx_te)[0]
    
    count_de = len(de_indices)
    target = len(te_indices) if target_count is None else target_count
    num_to_add = target - count_de
    
    print(f"\n[Oversampling] Class 169de current count: {count_de}, Target count (parity with 211Te): {target}")
    print(f"[Oversampling] Generating {num_to_add} augmented samples for 169de...")
    
    if num_to_add <= 0:
        return X_train, y_train
        
    aug_layer = keras.Sequential([
        layers.RandomRotation(0.05, seed=seed),
        layers.RandomTranslation(0.1, 0.1, seed=seed),
        layers.RandomZoom(0.1, seed=seed),
    ])
    
    rng = np.random.RandomState(seed)
    chosen_indices = rng.choice(de_indices, size=num_to_add, replace=True)
    de_samples = X_train[chosen_indices]
    
    augmented_samples = aug_layer(de_samples, training=True).numpy()
    augmented_labels = np.full(num_to_add, idx_de, dtype=y_train.dtype)
    
    X_aug = np.concatenate([X_train, augmented_samples], axis=0)
    y_aug = np.concatenate([y_train, augmented_labels], axis=0)
    
    shuffle_idx = rng.permutation(len(y_aug))
    print(f"[Oversampling] New training set shape: {X_aug.shape}, 169de count is now {np.sum(y_aug == idx_de)}")
    return X_aug[shuffle_idx], y_aug[shuffle_idx]


def main(model_type="dense", class_weighted=False, oversample_de=False):
    X_train = np.load("data/X_train.npy").reshape(-1, 28, 28, 1)
    y_train = np.load("data/y_train.npy")
    X_val = np.load("data/X_val.npy").reshape(-1, 28, 28, 1)
    y_val = np.load("data/y_val.npy")
    X_test = np.load("data/X_test.npy").reshape(-1, 28, 28, 1)
    y_test = np.load("data/y_test.npy")
    class_names = np.load("data/class_names.npy")

    if oversample_de:
        X_train, y_train = oversample_class_de(X_train, y_train, class_names, seed=SEED)

    if oversample_de:
        suffix = "_oversampled"
    elif class_weighted:
        suffix = "_weighted"
    else:
        suffix = ""

    if model_type == "cnn":
        model = build_cnn_model()
        model_path = f"models/amharic_char_model_cnn{suffix}.keras"
        history_path = f"results/training_history_cnn{suffix}.json"
    else:
        model = build_dense_model()
        model_path = f"models/amharic_char_model{suffix}.keras"
        history_path = f"results/training_history{suffix}.json"

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # Compute balanced class weights if requested
    class_weights_dict = None
    if class_weighted:
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weights_dict = {int(c): float(w) for c, w in zip(classes, weights)}
        print("\nClass weighting enabled (balanced inverse class frequency).")
        print(f"Sample weights: class 23 (169de): {class_weights_dict.get(23, 1.0):.4f}, class 29 (211Te): {class_weights_dict.get(29, 1.0):.4f}")

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        class_weight=class_weights_dict,
        callbacks=[early_stop],
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal test accuracy: {test_acc:.4f}")
    print(f"Final test loss: {test_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    model.save(model_path)

    os.makedirs("results", exist_ok=True)
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_dict["test_accuracy"] = float(test_acc)
    history_dict["test_loss"] = float(test_loss)
    if class_weighted:
        history_dict["class_weights"] = class_weights_dict
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Amharic character recognition model")
    parser.add_argument(
        "--model",
        type=str,
        choices=["dense", "cnn"],
        default="dense",
        help="Model architecture to train (dense or cnn, default: dense)",
    )
    parser.add_argument(
        "--class-weighted",
        action="store_true",
        help="Enable balanced inverse class frequency weighting during training",
    )
    parser.add_argument(
        "--oversample-de",
        action="store_true",
        help="Enable targeted data augmentation oversampling for class 169de (ደ) up to parity with 211Te",
    )
    args = parser.parse_args()
    main(model_type=args.model, class_weighted=args.class_weighted, oversample_de=args.oversample_de)
