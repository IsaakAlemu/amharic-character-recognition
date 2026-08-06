"""
Load the 33 base-character Amharic image folders, preprocess, and split
into train/val/test sets. Also trains a logistic regression baseline.

Usage:
    python src/preprocess.py --data_dir path/to/amharic_base34
"""

import argparse
import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

IMG_SIZE = 28
SEED = 42


def load_dataset(data_dir):
    classes = sorted(os.listdir(data_dir))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    X, y = [], []
    for c in classes:
        class_dir = os.path.join(data_dir, c)
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".jpeg")):
                continue
            img = Image.open(os.path.join(class_dir, fname)).convert("L")
            if img.size != (IMG_SIZE, IMG_SIZE):
                img = img.resize((IMG_SIZE, IMG_SIZE))
            arr = np.array(img, dtype=np.float32) / 255.0
            X.append(arr)
            y.append(class_to_idx[c])

    return np.stack(X), np.array(y), classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/amharic_base34")
    parser.add_argument("--out_dir", default="data")
    args = parser.parse_args()

    print(f"Loading images from {args.data_dir} ...")
    X, y, classes = load_dataset(args.data_dir)
    print(f"Loaded {X.shape[0]} images across {len(classes)} classes")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )
    print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(args.out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.out_dir, "X_val.npy"), X_val)
    np.save(os.path.join(args.out_dir, "y_val.npy"), y_val)
    np.save(os.path.join(args.out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(args.out_dir, "y_test.npy"), y_test)
    np.save(os.path.join(args.out_dir, "class_names.npy"), np.array(classes))
    print(f"Saved processed arrays to {args.out_dir}/")

    print("\nTraining baseline (logistic regression on raw pixels)...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_flat, y_train)
    val_acc = accuracy_score(y_val, clf.predict(X_val_flat))
    print(f"Baseline validation accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
