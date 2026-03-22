<div align="center">

<!-- Drop your photo into assets/profile.jpg and it will appear here -->
<img src="assets/profile.jpg" alt="Profile Picture" width="120" style="border-radius:50%;" />

# MNIST Handwritten Digit Detector

**An incremental series of CNN experiments on MNIST — starting from a clean baseline and progressively applying improvement techniques.**

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)](https://pytorch.org/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-98.68%25-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## Overview

This project classifies handwritten digits (0–9) from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) using CNNs. Each step builds on the previous one — new techniques are added incrementally so the impact of each change can be measured clearly.

**Common pipeline (all steps):**
- Explicit **train / validation / test** splits (54 k / 6 k / 10 k)
- **Early stopping** with best-model restoration
- Per-epoch training and validation metrics
- Full **evaluation**: confusion matrix, precision, recall, F1-score per class

---

## Experiments

| Step | Notebook | Key Change | Test Accuracy |
|------|----------|-----------|---------------|
| **Step 0** — Linear Baseline | [`step0_linear_without_hidden copy.ipynb`](step0_linear_without_hidden%20copy.ipynb) | Single linear layer | **91.49 %** |
| **Step 1** — MLP Baseline | [`step1_linear_MLP.ipynb`](step1_linear_MLP.ipynb) | MLP with hidden layers | **96.14 %** |
| **Step 2** — Simple CNN | [`step2_simpleCNN.ipynb`](step2_simpleCNN.ipynb) | Convolutional layers | **98.60 %** |

---

## Step 0 — Linear Baseline

> **Notebook:** [`step0_linear_without_hidden copy.ipynb`](step0_linear_without_hidden%20copy.ipynb)

A simple linear classifier without any hidden layers, serving as the baseline for comparison.

### Architecture

```
SimpleMLP
└── Linear(784 → 10)  [logits]
```

**Hyperparameters**

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Batch size | 32 |
| Epochs | 1 |
| Dropout | N/A |

### Test Results

| Metric | Value |
|---|---|
| Test Loss | **0.2912** |
| Test Accuracy | **91.49 %** |

## Step 1 — MLP Baseline

> **Notebook:** [`step1_linear_MLP.ipynb`](step1_linear_MLP.ipynb)

A multi-layer perceptron with a single hidden layer to improve upon the linear baseline.

### Architecture

```
SimpleMLP
└── Linear(784 → 10)  [logits]
```

**Hyperparameters**

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Batch size | 32 |
| Epochs | 1 |
| Dropout | N/A |

### Test Results

| Metric | Value |
|---|---|
| Test Loss | **0.1186** |
| Test Accuracy | **96.14 %** |

## Step 2 — Simple CNN

> **Notebook:** [`step2_simpleCNN.ipynb`](step2_simpleCNN.ipynb)

A simple convolutional neural network with two convolutional layers followed by fully connected layers.

### Architecture

```
SimpleCNN
├── Conv2d(1 → 10, kernel=5)  + ReLU + MaxPool2d(2)
├── Conv2d(10 → 20, kernel=5) + ReLU + MaxPool2d(2)
├── Flatten → Linear(320 → 50) + ReLU
└── Linear(50 → 10)           [logits]
```

**Hyperparameters**

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Batch size | 128 |
| Epochs | 5 (early stop patience=2) |
| Dropout | N/A |

### Training History

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 01 | 0.4070 | 88.12 % | 0.1306 | 96.15 % |
| 02 | 0.1072 | 96.65 % | 0.0881 | 97.37 % |
| 03 | 0.0766 | 97.61 % | 0.0757 | 97.67 % |
| 04 | 0.0615 | 98.07 % | 0.0635 | 98.05 % |
| 05 | 0.0510 | 98.46 % | 0.0557 | **98.25 %** |

### Test Results

| Metric | Value |
|---|---|
| Test Loss | **0.0425** |
| Test Accuracy | **98.60 %** |
```

---

## Dataset Split

| Split | Size |
|---|---|
| Train | 54 000 |
| Validation | 6 000 |
| Test | 10 000 |

Train/Validation are split from MNIST's official 60 000 training set (90/10).  
Test uses MNIST's standard held-out 10 000 samples.

---

## Project Structure

```
mnist_number_detector/
├── step1_simpleCNN.ipynb   # Baseline CNN (98.68 % test accuracy)
│                           #   Cell 1: model definition + training pipeline
│                           #   Cell 2: test set verification
│                           #   Cell 3: confusion matrix heatmap
│                           #   Cell 4: misclassified sample viewer
├── step2_*.ipynb           # (coming soon)
├── step3_*.ipynb           # (coming soon)
├── assets/
│   └── profile.jpg         # Your profile picture (add this)
├── data/                   # MNIST raw files (auto-downloaded, git-ignored)
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) **or** pip

### Install & run

```bash
git clone https://github.com/<your-username>/mnist_number_detector.git
cd mnist_number_detector

# With uv
uv venv && uv pip install torch torchvision scikit-learn matplotlib seaborn pandas

# Then open the notebook
jupyter lab step1_simpleCNN.ipynb
```

Run all cells top-to-bottom. MNIST data is downloaded automatically on first run.

---

## Add Your Profile Picture

```bash
mkdir -p assets
cp ~/your-photo.jpg assets/profile.jpg
git add assets/profile.jpg
git commit -m "Add profile picture"
```

---

## License

This project is licensed under the [MIT License](LICENSE).
