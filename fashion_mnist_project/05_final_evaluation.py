import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

np.random.seed(42)
torch.manual_seed(42)

from utils import (load_data, get_flat_arrays, plot_confusion_matrix)
import importlib
NeuralNet_module = importlib.import_module('03_neural_net_pytorch')
cnn_module = importlib.import_module('04_cnn_pytorch')
NeuralNet = NeuralNet_module.NeuralNet
CNN = cnn_module.CNN

device = torch.device('cpu')


def main():
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/results', exist_ok=True)

    print("Loading data...")
    train_ds, val_ds, test_ds = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = get_flat_arrays(train_ds, val_ds, test_ds)

    # Combine train+val for final sklearn models
    X_trainval = np.concatenate([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    # PCA on combined train+val
    pca = PCA(n_components=50, random_state=42)
    X_trainval_pca = pca.fit_transform(X_trainval)
    X_test_pca = pca.transform(X_test)

    # Load best hyperparams from sweep results
    sklearn_sweep = pd.read_csv('outputs/results/sklearn_sweep.csv')
    knn_rows = sklearn_sweep[sklearn_sweep['model'] == 'kNN']
    best_knn_row = knn_rows.loc[knn_rows['val_acc'].idxmax()]
    best_k = int(best_knn_row['hyperparam'].split('=')[1])

    lr_rows = sklearn_sweep[sklearn_sweep['model'] == 'LogReg']
    best_lr_row = lr_rows.loc[lr_rows['val_acc'].idxmax()]
    best_C = float(best_lr_row['hyperparam'].split('=')[1])

    results = []

    # ---- kNN ----
    print(f"\nTraining kNN (k={best_k}) on train+val...")
    start = time.time()
    knn = KNeighborsClassifier(n_neighbors=best_k, n_jobs=-1)
    knn.fit(X_trainval_pca, y_trainval)
    knn_time = time.time() - start
    knn_test_preds = knn.predict(X_test_pca)
    knn_test_acc = accuracy_score(y_test, knn_test_preds)
    knn_val_acc = float(best_knn_row['val_acc'])
    print(f"  kNN test_acc: {knn_test_acc:.4f} (val_acc: {knn_val_acc:.4f})")
    results.append({
        'model': 'kNN', 'val_acc': knn_val_acc,
        'test_acc': knn_test_acc, 'train_time_sec': knn_time
    })

    # ---- Logistic Regression ----
    print(f"\nTraining LogReg (C={best_C}) on train+val...")
    start = time.time()
    lr = LogisticRegression(solver='saga', max_iter=1000, C=best_C,
                            n_jobs=-1, random_state=42)
    lr.fit(X_trainval_pca, y_trainval)
    lr_time = time.time() - start
    lr_test_preds = lr.predict(X_test_pca)
    lr_test_acc = accuracy_score(y_test, lr_test_preds)
    lr_val_acc = float(best_lr_row['val_acc'])
    print(f"  LogReg test_acc: {lr_test_acc:.4f} (val_acc: {lr_val_acc:.4f})")
    results.append({
        'model': 'Logistic Regression', 'val_acc': lr_val_acc,
        'test_acc': lr_test_acc, 'train_time_sec': lr_time
    })

    # ---- Neural Network ----
    print("\nLoading best Neural Network...")
    NeuralNet_config = pd.read_csv('outputs/results/best_NeuralNet_config.csv')
    hidden_size = int(NeuralNet_config['hidden_size'].iloc[0])
    NeuralNet_model = NeuralNet(hidden_size).to(device)
    NeuralNet_model.load_state_dict(torch.load('outputs/results/best_NeuralNet.pt', map_location=device))

    NeuralNet_sweep = pd.read_csv('outputs/results/NeuralNet_sweep.csv')
    NeuralNet_val_acc = NeuralNet_sweep['best_val_acc'].max()

    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    NeuralNet_model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = NeuralNet_model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    NeuralNet_test_acc = correct / total
    print(f"  NeuralNet test_acc: {NeuralNet_test_acc:.4f} (val_acc: {NeuralNet_val_acc:.4f})")
    results.append({
        'model': 'NeuralNet', 'val_acc': NeuralNet_val_acc,
        'test_acc': NeuralNet_test_acc, 'train_time_sec': 0
    })

    # ---- CNN ----
    print("\nLoading best CNN...")
    cnn_config = pd.read_csv('outputs/results/best_cnn_config.csv')
    filters1 = int(cnn_config['filters1'].iloc[0])
    filters2 = int(cnn_config['filters2'].iloc[0])
    cnn_model = CNN(filters1, filters2).to(device)
    cnn_model.load_state_dict(torch.load('outputs/results/best_cnn.pt', map_location=device))

    cnn_sweep = pd.read_csv('outputs/results/cnn_sweep.csv')
    cnn_val_acc = cnn_sweep['best_val_acc'].max()

    cnn_model.eval()
    correct = 0
    total = 0
    cnn_preds = []
    cnn_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = cnn_model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            cnn_preds.extend(predicted.cpu().numpy())
            cnn_labels.extend(labels.numpy())
    cnn_test_acc = correct / total
    print(f"  CNN test_acc: {cnn_test_acc:.4f} (val_acc: {cnn_val_acc:.4f})")
    results.append({
        'model': 'CNN', 'val_acc': cnn_val_acc,
        'test_acc': cnn_test_acc, 'train_time_sec': 0
    })

    # ---- Results Table ----
    print("\n" + "=" * 60)
    print(f"{'Model':<25} {'Val Acc':>10} {'Test Acc':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<25} {r['val_acc']:>10.4f} {r['test_acc']:>10.4f}")
    print("=" * 60)

    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_csv('outputs/results/final_results.csv', index=False)
    print("\nSaved: outputs/results/final_results.csv")

    # ---- Bar chart: val vs test accuracy ----
    fig, ax = plt.subplots(figsize=(10, 6))
    model_names = [r['model'] for r in results]
    val_accs = [r['val_acc'] for r in results]
    test_accs = [r['test_acc'] for r in results]
    x = np.arange(len(model_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, val_accs, width, label='Validation', color='steelblue')
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test', color='coral')
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison: Validation vs Test Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0.7, 1.0)
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    fontsize=8)
    plt.savefig('outputs/figures/results_barchart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/figures/results_barchart.png")

    # ---- CNN confusion matrix on test set ----
    plot_confusion_matrix(np.array(cnn_labels), np.array(cnn_preds),
                          'CNN Confusion Matrix (Test Set)',
                          'outputs/figures/cnn_confusion_test.png')
    print("Saved: outputs/figures/cnn_confusion_test.png")

    print("\nFinal evaluation complete!")


if __name__ == '__main__':
    main()
