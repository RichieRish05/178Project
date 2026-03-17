# Fashion-MNIST ML Project — Claude Code Plan

## Project Overview
Build a machine learning pipeline for Fashion-MNIST image classification, comparing kNN, Logistic Regression, MLP, and a small CNN. Goal: produce all results and figures needed for a 4–5 page academic report in 2 days on CPU.

---

## Directory Structure to Create

```
fashion_mnist_project/
├── data/                   # Auto-downloaded by torchvision
├── outputs/
│   ├── figures/            # All plots saved here
│   └── results/            # CSV files with accuracy scores
├── 01_data_exploration.py
├── 02_sklearn_models.py
├── 03_mlp_pytorch.py
├── 04_cnn_pytorch.py
├── 05_final_evaluation.py
└── utils.py
```

---

## Step-by-Step Tasks for Claude Code

### Task 1 — Create `utils.py`
Shared utilities used across all scripts.

**Contents:**
- `load_data()` — loads Fashion-MNIST via torchvision, returns train/val/test splits (48k/12k/10k), fixed `random_seed=42`
- `get_flat_arrays()` — flattens image tensors to numpy arrays for sklearn models
- `apply_pca(X_train, X_val, X_test, n_components=50)` — fits PCA on train, transforms all three splits
- `plot_confusion_matrix(y_true, y_pred, title, save_path)` — saves a labeled confusion matrix PNG using seaborn
- `save_results(model_name, val_acc, test_acc, train_time, filepath='outputs/results/results.csv')` — appends a row to the results CSV
- Class names list: `['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']`

---

### Task 2 — Create `01_data_exploration.py`
Generates all figures for the Data Description section of the report.

**Figures to produce and save to `outputs/figures/`:**

1. `sample_grid.png` — 5×10 grid showing 5 examples per class, with class name titles
2. `class_distribution.png` — bar chart of class counts in the training set
3. `mean_per_class.png` — 2×5 grid of mean images per class (computed across all 60k train examples)
4. `pixel_histogram.png` — histogram of pixel intensity values across the full training set

**Also print to console:**
- Dataset sizes (train/val/test)
- Min, max, mean pixel value
- Confirmation that classes are balanced

---

### Task 3 — Create `02_sklearn_models.py`
Trains and tunes kNN and Logistic Regression using PCA-reduced features.

**kNN:**
- Use `KNeighborsClassifier(n_jobs=-1)`
- Sweep `k` values: `[3, 5, 10, 15]`
- For each k: fit on train PCA, evaluate on val PCA, print val accuracy
- Select best k by val accuracy
- Save confusion matrix for best k to `outputs/figures/knn_confusion.png`

**Logistic Regression:**
- Use `LogisticRegression(solver='saga', max_iter=1000, n_jobs=-1, random_state=42)`
- Sweep `C` values: `[0.01, 0.1, 1.0, 10.0]`
- For each C: fit on train PCA, evaluate on val PCA, print val accuracy
- Select best C by val accuracy
- Save confusion matrix for best C to `outputs/figures/logreg_confusion.png`

**At the end of this script:**
- Print best hyperparameters for both models
- Save val accuracies for all hyperparameter values to `outputs/results/sklearn_sweep.csv`
- Do NOT evaluate on test set yet — that happens in Task 5

---

### Task 4 — Create `03_mlp_pytorch.py`
Trains a feedforward MLP with a small hyperparameter sweep.

**Architecture:**
```
Flatten → Linear(784, hidden) → ReLU → Dropout(p) → Linear(hidden, hidden) → ReLU → Dropout(p) → Linear(hidden, 10)
```

**Hyperparameter sweep** (4 combinations only — keep it fast):
| hidden_size | dropout | lr     |
|-------------|---------|--------|
| 256         | 0.3     | 1e-3   |
| 256         | 0.2     | 5e-4   |
| 128         | 0.3     | 1e-3   |
| 128         | 0.2     | 1e-3   |

**Training setup:**
- Optimizer: `Adam`
- Loss: `CrossEntropyLoss`
- Epochs: 15
- Batch size: 64
- Print train loss and val accuracy each epoch

**Outputs:**
- Save val accuracy curve plot for the best config to `outputs/figures/mlp_val_curve.png`
- Save confusion matrix for best config to `outputs/figures/mlp_confusion.png`
- Save all sweep results to `outputs/results/mlp_sweep.csv`
- Save best model weights to `outputs/results/best_mlp.pt`

---

### Task 5 — Create `04_cnn_pytorch.py`
Trains a small CNN — the most important model in the project.

**Architecture:**
```
Conv2d(1, 32, 3, padding=1) → ReLU → MaxPool2d(2)     # → 14×14
Conv2d(32, 64, 3, padding=1) → ReLU → MaxPool2d(2)    # → 7×7
Flatten → Linear(64*7*7, 128) → ReLU → Dropout(0.5) → Linear(128, 10)
```

**Hyperparameter sweep** (3 combinations):
| filters     | dropout | lr     |
|-------------|---------|--------|
| 32/64       | 0.5     | 1e-3   |
| 32/64       | 0.3     | 1e-3   |
| 16/32       | 0.5     | 1e-3   |

**Training setup:**
- Optimizer: `Adam`
- Loss: `CrossEntropyLoss`
- Epochs: 20
- Batch size: 64
- Print train loss and val accuracy each epoch

**Outputs:**
- Save val accuracy curve for best config to `outputs/figures/cnn_val_curve.png`
- Save confusion matrix for best config to `outputs/figures/cnn_confusion.png`
- Save all sweep results to `outputs/results/cnn_sweep.csv`
- Save best model weights to `outputs/results/best_cnn.pt`

---

### Task 6 — Create `05_final_evaluation.py`
Runs all 4 best models on the held-out test set exactly once.

**Steps:**
1. Load test set from `utils.py`
2. Load best kNN and Logistic Regression (re-fit using best hyperparams found in Task 3 on full train+val data for final eval)
3. Load best MLP weights from `outputs/results/best_mlp.pt`
4. Load best CNN weights from `outputs/results/best_cnn.pt`
5. Evaluate all 4 on test set
6. Print a formatted results table to console
7. Save final results to `outputs/results/final_results.csv` with columns: `model, val_acc, test_acc, train_time_sec`
8. Generate `outputs/figures/results_barchart.png` — grouped bar chart comparing val vs test accuracy for all 4 models
9. Generate `outputs/figures/cnn_confusion_test.png` — confusion matrix on test set for CNN only (best model)

---

## Key Implementation Notes for Claude Code

- **Random seed**: Set `torch.manual_seed(42)` and `numpy.random.seed(42)` at the top of every script
- **Device**: Force CPU — `device = torch.device('cpu')` — do not use `.cuda()`
- **Data loading**: Use `torchvision.datasets.FashionMNIST(root='./data', download=True)`
- **No test set leakage**: Scripts 02–04 never call anything with `X_test` or `y_test`. Only script 05 touches test data
- **Figure saving**: All figures use `plt.savefig(path, dpi=150, bbox_inches='tight')` and `plt.close()` immediately after — do not call `plt.show()`
- **Progress printing**: Each training script should print epoch-level progress so it's clear the script is running
- **Install requirements**: All scripts assume `torch`, `torchvision`, `scikit-learn`, `matplotlib`, `seaborn`, `numpy`, `pandas` are installed

---

## Expected Output Files for the Report

| File | Used in Report Section |
|------|----------------------|
| `figures/sample_grid.png` | Data Description (Figure 1) |
| `figures/mean_per_class.png` | Data Description (Figure 2) |
| `figures/class_distribution.png` | Data Description |
| `figures/results_barchart.png` | Experimental Results |
| `figures/cnn_confusion_test.png` | Experimental Results / Insights |
| `results/final_results.csv` | Results Table |
| `results/mlp_sweep.csv` | Appendix (hyperparameter sweep) |
| `results/cnn_sweep.csv` | Appendix (hyperparameter sweep) |

---

## Execution Order

```bash
python utils.py                  # verify no import errors
python 01_data_exploration.py    # ~1 min
python 02_sklearn_models.py      # ~10 min
python 03_mlp_pytorch.py         # ~45 min
python 04_cnn_pytorch.py         # ~60 min
python 05_final_evaluation.py    # ~5 min
```

Total estimated runtime on CPU laptop: **~2 hours**