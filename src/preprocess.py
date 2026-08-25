"""
Load Amharic character image folders/datasets (supporting both 33 base classes
and full 238-class 34x7 Fidel set), preprocess, and split into train/val/test sets.

Usage:
    python src/preprocess.py --data_dir data/amharic_all
    python src/preprocess.py --data_dir data/amharic_base34
"""

import argparse
import os
from collections import Counter
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_SIZE = 28
SEED = 42


def find_all_images(data_dir):
    """
    Recursively discover all image files in data_dir.
    Supports:
      1. Folder-per-class structure: data_dir/<class_name>/<img_file>.jpg
      2. File-prefix structure: data_dir/<dataset_folder>/<class_name>.<id>.jpg
    """
    image_entries = []
    
    # Check direct subdirectories
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # If data_dir has subdirectories, inspect their contents
    if subdirs:
        for s in subdirs:
            s_path = os.path.join(data_dir, s)
            for fname in os.listdir(s_path):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                # Determine class name
                if "." in fname:
                    cname = fname.split(".")[0]
                else:
                    cname = s
                # If folder itself is a class name (e.g. 001he)
                if s.isalnum() and len(s) >= 4 and s[:3].isdigit():
                    cname = s
                image_entries.append((os.path.join(s_path, fname), cname))
    else:
        # Flat directory
        for fname in os.listdir(data_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            cname = fname.split(".")[0]
            image_entries.append((os.path.join(data_dir, fname), cname))

    return image_entries


def load_dataset(data_dir):
    entries = find_all_images(data_dir)
    if not entries:
        raise ValueError(f"No valid image files found in {data_dir}")

    classes = sorted(list(set(c for _, c in entries)))
    class_to_idx = {c: i for i, c in enumerate(classes)}

    print(f"Discovered {len(entries)} image files across {len(classes)} classes.")

    X = np.empty((len(entries), IMG_SIZE, IMG_SIZE), dtype=np.float32)
    y = np.empty(len(entries), dtype=np.int32)

    for i, (path, cname) in enumerate(entries):
        with Image.open(path) as img:
            img_gray = img.convert("L")
            if img_gray.size != (IMG_SIZE, IMG_SIZE):
                img_gray = img_gray.resize((IMG_SIZE, IMG_SIZE))
            X[i] = np.array(img_gray, dtype=np.float32) / 255.0
            y[i] = class_to_idx[cname]

    return X, y, classes


def print_distribution_stats(classes, y):
    counts = Counter(y)
    class_counts = [(classes[i], counts[i]) for i in range(len(classes))]
    sample_sizes = [cnt for _, cnt in class_counts]

    min_cnt = min(sample_sizes)
    max_cnt = max(sample_sizes)
    avg_cnt = np.mean(sample_sizes)
    median_cnt = np.median(sample_sizes)

    print("\n" + "=" * 60)
    print("Dataset Class Distribution Summary:")
    print(f"  Total Classes:   {len(classes)}")
    print(f"  Total Images:    {len(y):,}")
    print(f"  Min Samples:     {min_cnt} ({[c for c, cnt in class_counts if cnt == min_cnt]})")
    print(f"  Max Samples:     {max_cnt} ({[c for c, cnt in class_counts if cnt == max_cnt]})")
    print(f"  Average Samples: {avg_cnt:.2f} per class")
    print(f"  Median Samples:  {median_cnt:.1f} per class")
    print("=" * 60)

    # Identify classes with notably fewer samples (e.g. < 100 samples)
    low_sample_classes = [(c, cnt) for c, cnt in class_counts if cnt < 100]
    low_sample_classes.sort(key=lambda x: x[1])

    print(f"\nClasses with notably fewer samples (< 100 images, {len(low_sample_classes)} total):")
    for c, cnt in low_sample_classes:
        pct_of_avg = (cnt / avg_cnt) * 100
        print(f"  {c:8s}: {cnt:3d} images ({pct_of_avg:4.1f}% of dataset average)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/amharic_all")
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--train_baseline", action="store_true", help="Train logistic regression baseline")
    args = parser.parse_args()

    print(f"Loading images from {args.data_dir} ...")
    X, y, classes = load_dataset(args.data_dir)
    
    print_distribution_stats(classes, y)

    # 70% Train, 15% Val, 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED
    )
    print(f"\nSplits created (70% / 15% / 15%):")
    print(f"  Train: {X_train.shape[0]:,} images")
    print(f"  Val:   {X_val.shape[0]:,} images")
    print(f"  Test:  {X_test.shape[0]:,} images")

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(args.out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.out_dir, "X_val.npy"), X_val)
    np.save(os.path.join(args.out_dir, "y_val.npy"), y_val)
    np.save(os.path.join(args.out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(args.out_dir, "y_test.npy"), y_test)
    np.save(os.path.join(args.out_dir, "class_names.npy"), np.array(classes))
    print(f"Saved processed arrays to {args.out_dir}/")

    if args.train_baseline:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        print("\nTraining baseline (logistic regression on raw pixels)...")
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train_flat, y_train)
        val_acc = accuracy_score(y_val, clf.predict(X_val_flat))
        print(f"Baseline validation accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
