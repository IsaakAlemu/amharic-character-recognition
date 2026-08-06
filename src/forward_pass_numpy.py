"""
A from-scratch NumPy implementation of a forward pass through a small
dense neural network (784 -> 64 -> 33), applied to real training data.

This exists to demonstrate the underlying math that TensorFlow's Dense
layers abstract away in src/train.py -- same computation, done by hand.
No training happens here, just a single forward pass on untrained,
randomly-initialized weights.

Usage:
    python src/forward_pass_numpy.py
"""

import numpy as np

SEED = 42


def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def main():
    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")

    X_flat = X_train.reshape(X_train.shape[0], -1)
    print(f"Input shape after flattening: {X_flat.shape}")

    np.random.seed(SEED)
    input_size, hidden_size, output_size = 784, 64, 33

    # Layer 1: hidden layer
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros(hidden_size)
    Z1 = X_flat @ W1 + b1
    A1 = np.maximum(0, Z1)  # ReLU

    # Layer 2: output layer
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros(output_size)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)

    print(f"\nLayer 1 weight shape: {W1.shape}  (expect {input_size}x{hidden_size})")
    print(f"Layer 2 weight shape: {W2.shape}  (expect {hidden_size}x{output_size})")
    print(f"\nFirst training image -- predicted probability distribution over 33 classes:")
    print(A2[0])
    print(f"\nSum of probabilities: {A2[0].sum():.6f}  (should be ~1.0)")
    print(f"Predicted class (untrained, essentially random): {A2[0].argmax()}")
    print(
        "\nNote: with random, untrained weights, probabilities are expected "
        "to be near-uniform (~1/33 each) -- the model has no learned basis "
        "for preferring any class yet. This confirms the forward-pass math "
        "is correct, independent of training."
    )


if __name__ == "__main__":
    main()
