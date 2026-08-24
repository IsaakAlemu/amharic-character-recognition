# Amharic Handwritten Character Recognition

A neural network that classifies handwritten Amharic characters, built to
practice neural network fundamentals (forward propagation, dense layers,
softmax classification) while tackling a real, underserved problem: Amharic
handwriting recognition, an area with far less tooling and research
attention than Latin-script OCR.

## Why this scope

Amharic's full script (Ge'ez/Fidel) has 33 base consonants, each written in
7 vowel-modified forms ("orders") — 231+ total character shapes. This
project deliberately scopes down to the **33 base (first-order) characters
only**. This keeps the model, a plain dense neural network, appropriately
matched to the problem's complexity, while still tackling a real
classification task grounded in actual research datasets rather than a toy
problem like MNIST.

## Results

| Model | Test Accuracy |
|---|---|
| Logistic Regression (baseline) | 82.6% |
| Dense Neural Network | 86.3% |
| **Convolutional Neural Network (CNN)** | **95.8%** |

![Baseline comparison](results/baseline_comparison.png)

The dense neural network achieved an 86.3% test accuracy (~3.6 percentage
points over the linear baseline). Upgrading to a CNN yielded a substantial leap
to **95.77% test accuracy** (+9.5 pp over Dense NN, +13.1 pp over baseline).
Across 3 runs with different random seeds, CNN test accuracy consistently
ranged from **95.4% to 96.5%** (averaging **95.89%**), confirming the stability
of spatial feature learning across initializations.

### Training curves

![Training curves](results/training_curves.png)

Validation accuracy tracks *above* training accuracy for most of training —
a signature of the data augmentation (random rotation/translation/zoom)
working as intended: it makes the training task harder, which reduces
overfitting rather than causing it.

### Confusion matrix

![Confusion matrix](results/confusion_matrix.png)

Predictions are strongly concentrated on the diagonal (correct), with a
small number of specific, consistent confusions rather than widespread
errors.

**The most significant confusion, by a wide margin:** ደ (`169de`) predicted
as ጠ (`211Te`):
- **Dense NN**: 12 of 21 test instances of ደ (57.1% error rate, 5 of 21 correct)
- **CNN**: 10 of 21 test instances (47.6% error rate, a 9.5 percentage-point reduction) — doubling correct classifications from 5 to 10, though this pair remains the model's single largest confusion even after the architecture upgrade.

These are two visually similar but phonetically distinct letters — a plain "d" versus an ejective "t'" sound — and this pair is also a commonly-confused pair for human learners of
Amharic script. The model's biggest mistake tracks a genuine visual
ambiguity in the writing system itself, not an arbitrary pixel-level quirk.

### Sample predictions

![Sample predictions](results/sample_predictions.png)

## Approach

1. **Data**: [Handwritten Amharic character dataset](https://github.com/Fetulhak/Handwritten-Amharic-character-Dataset)
   (Fetulhak), filtered to the 33 base-order characters — 5,678 images,
   28×28 grayscale, ~172 images/class average.
2. **Forward propagation from scratch** (`src/forward_pass_numpy.py`) — a
   hand-written NumPy implementation of the exact dense-layer math
   (`X · W + b`, ReLU, softmax), verified against course exercises, before
   touching any framework.
3. **Baseline**: logistic regression on flattened pixels, to establish
   what a linear model can already do.
4. **Neural network** (`src/train.py`): `Dense(128, relu) → Dense(64, relu)
   → Dense(33, softmax)`, with on-the-fly data augmentation (rotation,
   translation, zoom) and early stopping on validation accuracy.
5. **Evaluation** (`src/evaluate.py`): confusion matrix, baseline
   comparison, and misclassification analysis.

## Project structure

```
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── src/
│   ├── preprocess.py           # Load images, split data, train baseline
│   ├── forward_pass_numpy.py   # Hand-written forward pass (no framework)
│   ├── train.py                # Build, train, and save the final model
│   └── evaluate.py             # Generate all metrics and visuals
├── results/                    # Generated metrics, plots, confusion matrix
├── models/                     # Saved trained model
└── data/                       # Processed .npy arrays (raw images not included)
```

## Running it

```bash
pip install -r requirements.txt

# 1. Preprocess (expects raw images in data/amharic_base34/<class>/*.jpg)
python src/preprocess.py --data_dir data/amharic_base34

# 2. See the forward pass math in isolation
python src/forward_pass_numpy.py

# 3. Train the final model
python src/train.py

# 4. Generate evaluation visuals
python src/evaluate.py
```

## Data source & license note

Raw images are not redistributed in this repository. The dataset is
publicly available at the link above; see its own repository for licensing
terms. This project's code is released under the license in `LICENSE`.

## Future extensions

- Extend to all ~240 character forms (all 7 vowel orders, not just base)
- Targeted disambiguation for ደ/ጠ — this pair remains an open bottleneck even with a CNN, suggesting it may require targeted strategies (such as supplemental training data or attention-based fine-grained feature extraction) rather than broader architecture shifts
- Move from isolated-character to word/sentence-level recognition
- Build a small demo (upload an image, get back predicted text)
