import numpy as np
import torch
import time
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(42)
torch.manual_seed(42)

from utils import load_data, get_flat_arrays, apply_pca, plot_confusion_matrix

def main():
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/results', exist_ok=True)

    print("Loading data...")
    train_ds, val_ds, test_ds = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = get_flat_arrays(train_ds, val_ds, test_ds)

    print("Applying PCA (50 components)...")
    X_train_pca, X_val_pca, _ = apply_pca(X_train, X_val, X_test)

    sweep_results = []

    # ---- kNN ----
    print("\n=== kNN Hyperparameter Sweep ===")
    best_knn_acc = 0
    best_k = None
    best_knn_preds = None

    for k in [3, 5, 10, 15]:
        print(f"  k={k}: ", end="", flush=True)
        start = time.time()
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(X_train_pca, y_train)
        elapsed = time.time() - start
        val_preds = knn.predict(X_val_pca)
        val_acc = accuracy_score(y_val, val_preds)
        print(f"val_acc={val_acc:.4f} ({elapsed:.1f}s)")
        sweep_results.append({
            'model': 'kNN', 'hyperparam': f'k={k}',
            'val_acc': val_acc, 'train_time_sec': elapsed
        })
        if val_acc > best_knn_acc:
            best_knn_acc = val_acc
            best_k = k
            best_knn_preds = val_preds

    print(f"  Best k: {best_k} (val_acc={best_knn_acc:.4f})")
    plot_confusion_matrix(y_val, best_knn_preds,
                          f'kNN (k={best_k}) Confusion Matrix (Validation)',
                          'outputs/figures/knn_confusion.png')
    print("  Saved: outputs/figures/knn_confusion.png")

    # ---- Logistic Regression ----
    print("\n=== Logistic Regression Hyperparameter Sweep ===")
    best_lr_acc = 0
    best_C = None
    best_lr_preds = None

    # Test different regularization strengths
    for C in [0.01, 0.1, 1.0, 10.0]:
        print(f"  C={C}: ", end="", flush=True)
        start = time.time()
        lr = LogisticRegression(solver='saga', max_iter=1000, C=C,
                                n_jobs=-1, random_state=42)
        lr.fit(X_train_pca, y_train)
        elapsed = time.time() - start
        val_preds = lr.predict(X_val_pca)
        val_acc = accuracy_score(y_val, val_preds)
        print(f"val_acc={val_acc:.4f} ({elapsed:.1f}s)")
        sweep_results.append({
            'model': 'LogReg', 'hyperparam': f'C={C}',
            'val_acc': val_acc, 'train_time_sec': elapsed
        })
        if val_acc > best_lr_acc:
            best_lr_acc = val_acc
            best_C = C
            best_lr_preds = val_preds

    print(f"  Best C: {best_C} (val_acc={best_lr_acc:.4f})")
    plot_confusion_matrix(y_val, best_lr_preds,
                          f'Logistic Regression (C={best_C}) Confusion Matrix (Validation)',
                          'outputs/figures/logreg_confusion.png')
    print("  Saved: outputs/figures/logreg_confusion.png")

    # Save sweep results
    pd.DataFrame(sweep_results).to_csv('outputs/results/sklearn_sweep.csv', index=False)
    print("\nSaved: outputs/results/sklearn_sweep.csv")

    print(f"\n=== Summary ===")
    print(f"Best kNN: k={best_k}, val_acc={best_knn_acc:.4f}")
    print(f"Best LogReg: C={best_C}, val_acc={best_lr_acc:.4f}")
    print("(Test evaluation deferred to 05_final_evaluation.py)")


if __name__ == '__main__':
    main()
