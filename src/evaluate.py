"""
Generate all evaluation visuals for the README: training curves,
confusion matrix, baseline comparison, and sample predictions.

Usage:
    python src/evaluate.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
import os

os.makedirs("results", exist_ok=True)

# --- Load everything ---
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")
class_names = np.load("data/class_names.npy")

model = keras.models.load_model("models/amharic_char_model.keras")
with open("results/training_history.json") as f:
    history = json.load(f)

X_test_img = X_test.reshape(-1, 28, 28, 1)

# --- 1. Training curves ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(history["accuracy"], label="Train")
axes[0].plot(history["val_accuracy"], label="Validation")
axes[0].set_title("Accuracy over Training")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history["loss"], label="Train")
axes[1].plot(history["val_loss"], label="Validation")
axes[1].set_title("Loss over Training")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/training_curves.png", dpi=150)
plt.close()
print("Saved results/training_curves.png")

# --- 2. Baseline comparison bar chart ---
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
baseline = LogisticRegression(max_iter=1000)
baseline.fit(X_train_flat, y_train)
baseline_acc = accuracy_score(y_test, baseline.predict(X_test_flat))
nn_acc = history["test_accuracy"]

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(
    ["Logistic Regression\n(baseline)", "Neural Network\n(this project)"],
    [baseline_acc * 100, nn_acc * 100],
    color=["#94a3b8", "#2563eb"],
)
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("Baseline vs. Neural Network")
ax.set_ylim(0, 100)
for bar, val in zip(bars, [baseline_acc * 100, nn_acc * 100]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.1f}%",
            ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("results/baseline_comparison.png", dpi=150)
plt.close()
print(f"Saved results/baseline_comparison.png (baseline={baseline_acc:.4f}, nn={nn_acc:.4f})")

# --- 3. Confusion matrix ---
y_pred = np.argmax(model.predict(X_test_img, verbose=0), axis=1)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(13, 11))
sns.heatmap(cm, cmap="Blues", xticklabels=class_names, yticklabels=class_names,
            cbar_kws={"label": "Count"})
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - 33 Base Amharic Characters")
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
plt.close()
print("Saved results/confusion_matrix.png")

# --- 4. Top confusions (text) ---
cm_copy = cm.copy()
np.fill_diagonal(cm_copy, 0)
flat_indices = np.argsort(cm_copy.flatten())[::-1][:10]
top_confusions = []
for idx in flat_indices:
    true_idx, pred_idx = np.unravel_index(idx, cm_copy.shape)
    count = cm_copy[true_idx, pred_idx]
    if count > 0:
        top_confusions.append({
            "true": str(class_names[true_idx]),
            "predicted": str(class_names[pred_idx]),
            "count": int(count),
        })
with open("results/top_confusions.json", "w") as f:
    json.dump(top_confusions, f, indent=2)
print("Saved results/top_confusions.json")
for c in top_confusions[:5]:
    print(f"  {c['true']} -> {c['predicted']}: {c['count']} times")

# --- 5. Sample predictions grid (mix of correct and incorrect, for honesty) ---
rng = np.random.RandomState(SEED := 42)
correct_mask = y_pred == y_test
incorrect_idx = np.where(~correct_mask)[0]
correct_idx = np.where(correct_mask)[0]
n_incorrect = min(4, len(incorrect_idx))
sample_idx = np.concatenate([
    rng.choice(incorrect_idx, n_incorrect, replace=False),
    rng.choice(correct_idx, 12 - n_incorrect, replace=False),
])
rng.shuffle(sample_idx)
fig, axes = plt.subplots(3, 4, figsize=(10, 8))
for ax, i in zip(axes.flat, sample_idx):
    ax.imshow(X_test[i], cmap="gray")
    true_label = class_names[y_test[i]]
    pred_label = class_names[y_pred[i]]
    correct = y_test[i] == y_pred[i]
    color = "green" if correct else "red"
    ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=9)
    ax.axis("off")
plt.suptitle("Sample Test Predictions (green=correct, red=incorrect)", fontsize=12)
plt.tight_layout()
plt.savefig("results/sample_predictions.png", dpi=150)
plt.close()
print("Saved results/sample_predictions.png")

# --- Final metrics summary ---
summary = {
    "baseline_test_accuracy": float(baseline_acc),
    "neural_net_test_accuracy": float(nn_acc),
    "improvement_percentage_points": float((nn_acc - baseline_acc) * 100),
    "num_classes": len(class_names),
    "num_test_samples": len(y_test),
}
with open("results/metrics_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSummary:", json.dumps(summary, indent=2))
