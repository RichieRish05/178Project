import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import pandas as pd
import matplotlib.pyplot as plt
import os

np.random.seed(42)
torch.manual_seed(42)

from utils import load_data, plot_confusion_matrix

device = torch.device('cpu')


class MLP(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 10)
        )

    def forward(self, x):
        return self.net(x)


def train_and_evaluate(model, train_loader, val_loader, epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    val_accs = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        val_accs.append(val_acc)
        print(f"    Epoch {epoch+1:2d}/{epochs}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

    return val_accs


def get_predictions(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def main():
    os.makedirs('outputs/figures', exist_ok=True)
    os.makedirs('outputs/results', exist_ok=True)

    print("Loading data...")
    train_ds, val_ds, _ = load_data()
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    configs = [
        {'hidden_size': 256, 'dropout': 0.3, 'lr': 1e-3},
        {'hidden_size': 256, 'dropout': 0.2, 'lr': 5e-4},
        {'hidden_size': 128, 'dropout': 0.3, 'lr': 1e-3},
        {'hidden_size': 128, 'dropout': 0.2, 'lr': 1e-3},
    ]

    epochs = 15
    sweep_results = []
    best_val_acc = 0
    best_model = None
    best_val_accs = None
    best_config = None

    for i, cfg in enumerate(configs):
        print(f"\n=== Config {i+1}/{len(configs)}: hidden={cfg['hidden_size']}, "
              f"dropout={cfg['dropout']}, lr={cfg['lr']} ===")

        torch.manual_seed(42)
        model = MLP(cfg['hidden_size'], cfg['dropout']).to(device)

        start = time.time()
        val_accs = train_and_evaluate(model, train_loader, val_loader, epochs, cfg['lr'])
        elapsed = time.time() - start

        final_val_acc = val_accs[-1]
        max_val_acc = max(val_accs)
        print(f"    Final val_acc: {final_val_acc:.4f}, Best val_acc: {max_val_acc:.4f}, "
              f"Time: {elapsed:.1f}s")

        sweep_results.append({
            'hidden_size': cfg['hidden_size'],
            'dropout': cfg['dropout'],
            'lr': cfg['lr'],
            'best_val_acc': max_val_acc,
            'final_val_acc': final_val_acc,
            'train_time_sec': elapsed
        })

        if max_val_acc > best_val_acc:
            best_val_acc = max_val_acc
            best_model = model
            best_val_accs = val_accs
            best_config = cfg

    # Save val accuracy curve for best config
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, epochs + 1), best_val_accs, marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title(f"MLP Validation Accuracy (hidden={best_config['hidden_size']}, "
                 f"dropout={best_config['dropout']}, lr={best_config['lr']})")
    ax.grid(True, alpha=0.3)
    plt.savefig('outputs/figures/mlp_val_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: outputs/figures/mlp_val_curve.png")

    # Confusion matrix for best config
    y_true, y_pred = get_predictions(best_model, val_loader)
    plot_confusion_matrix(y_true, y_pred,
                          f"MLP Confusion Matrix (Validation)",
                          'outputs/figures/mlp_confusion.png')
    print("Saved: outputs/figures/mlp_confusion.png")

    # Save sweep results
    pd.DataFrame(sweep_results).to_csv('outputs/results/mlp_sweep.csv', index=False)
    print("Saved: outputs/results/mlp_sweep.csv")

    # Save best model
    torch.save(best_model.state_dict(), 'outputs/results/best_mlp.pt')
    print("Saved: outputs/results/best_mlp.pt")

    # Save best config info for final evaluation
    best_config_info = {
        'hidden_size': best_config['hidden_size'],
        'dropout': best_config['dropout']
    }
    pd.DataFrame([best_config_info]).to_csv('outputs/results/best_mlp_config.csv', index=False)

    print(f"\n=== Best MLP Config ===")
    print(f"hidden_size={best_config['hidden_size']}, dropout={best_config['dropout']}, "
          f"lr={best_config['lr']}")
    print(f"Best val_acc: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()
