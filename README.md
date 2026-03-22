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
| **Step 1** — Baseline SimpleCNN | [`step1_simpleCNN.ipynb`](step1_simpleCNN.ipynb) | Conv×2 + FC×2 + Dropout | **98.68 %** |
| Step 2 | *(coming soon)* | — | — |
| Step 3 | *(coming soon)* | — | — |

---

## Step 1 — Baseline SimpleCNN

> **Notebook:** [`step1_simpleCNN.ipynb`](step1_simpleCNN.ipynb)

A minimal two-conv-layer CNN trained with standard hyperparameters to establish a reliable baseline.

### Architecture

```
SimpleCNN
├── Conv2d(1 → 10, kernel=5)  + ReLU + MaxPool2d(2)
├── Conv2d(10 → 20, kernel=5) + ReLU + MaxPool2d(2)
├── Flatten → Linear(320 → 50) + ReLU + Dropout(0.2)
└── Linear(50 → 10)           [logits]
```

**Hyperparameters**

| Parameter | Value |
|---|---|
| Optimiser | Adam |
| Learning rate | 1e-3 |
| Batch size | 128 |
| Epochs | 5 (early stop patience=2) |
| Dropout | 0.2 |

### Training History

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|-----------|-----------|----------|---------|
| 01 | 0.3874 | 88.00 % | 0.1031 | 97.13 % |
| 02 | 0.1098 | 96.73 % | 0.0715 | 97.82 % |
| 03 | 0.0816 | 97.46 % | 0.0640 | 98.03 % |
| 04 | 0.0655 | 97.94 % | 0.0532 | 98.23 % |
| 05 | 0.0577 | 98.31 % | 0.0520 | **98.38 %** |

### Test Results

| Metric | Value |
|---|---|
| Test Loss | **0.0433** |
| Test Accuracy | **98.68 %** |

### Classification Report

```
              precision    recall  f1-score   support

           0     0.9809    0.9949    0.9878       980
           1     0.9930    0.9947    0.9938      1135
           2     0.9771    0.9922    0.9846      1032
           3     0.9910    0.9822    0.9866      1010
           4     0.9908    0.9868    0.9888       982
           5     0.9910    0.9843    0.9876       892
           6     0.9947    0.9864    0.9906       958
           7     0.9733    0.9922    0.9827      1028
           8     0.9917    0.9784    0.9850       974
           9     0.9860    0.9742    0.9801      1009

    accuracy                         0.9868     10000
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
