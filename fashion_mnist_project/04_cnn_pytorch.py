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


class CNN(nn.Module):
    def __init__(self, filters1, filters2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, filters1, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(filters1, filters2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(filters2 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


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
        {'filters1': 32, 'filters2': 64, 'lr': 1e-3},
        {'filters1': 16, 'filters2': 32, 'lr': 1e-3},
    ]

    epochs = 20
    sweep_results = []
    best_val_acc = 0
    best_model = None
    best_val_accs = None
    best_config = None

    for i, cfg in enumerate(configs):
        print(f"\n=== Config {i+1}/{len(configs)}: filters={cfg['filters1']}/{cfg['filters2']}, lr={cfg['lr']} ===")

        torch.manual_seed(42)
        model = CNN(cfg['filters1'], cfg['filters2']).to(device)

        start = time.time()
        val_accs = train_and_evaluate(model, train_loader, val_loader, epochs, cfg['lr'])
        elapsed = time.time() - start

        final_val_acc = val_accs[-1]
        max_val_acc = max(val_accs)
        print(f"    Final val_acc: {final_val_acc:.4f}, Best val_acc: {max_val_acc:.4f}, "
              f"Time: {elapsed:.1f}s")

        sweep_results.append({
            'filters': f"{cfg['filters1']}/{cfg['filters2']}",
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
    ax.set_title(f"CNN Validation Accuracy (filters={best_config['filters1']}/{best_config['filters2']})")
    ax.grid(True, alpha=0.3)
    plt.savefig('outputs/figures/cnn_val_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: outputs/figures/cnn_val_curve.png")

    # Confusion matrix for best config
    y_true, y_pred = get_predictions(best_model, val_loader)
    plot_confusion_matrix(y_true, y_pred,
                          f"CNN Confusion Matrix (Validation)",
                          'outputs/figures/cnn_confusion.png')
    print("Saved: outputs/figures/cnn_confusion.png")

    # Save sweep results
    pd.DataFrame(sweep_results).to_csv('outputs/results/cnn_sweep.csv', index=False)
    print("Saved: outputs/results/cnn_sweep.csv")

    # Save best model
    torch.save(best_model.state_dict(), 'outputs/results/best_cnn.pt')
    print("Saved: outputs/results/best_cnn.pt")

    # Save best config info for final evaluation
    best_config_info = {
        'filters1': best_config['filters1'],
        'filters2': best_config['filters2'],
    }
    pd.DataFrame([best_config_info]).to_csv('outputs/results/best_cnn_config.csv', index=False)

    print(f"\n=== Best CNN Config ===")
    print(f"filters={best_config['filters1']}/{best_config['filters2']}, lr={best_config['lr']}")
    print(f"Best val_acc: {best_val_acc:.4f}")


if __name__ == '__main__':
    main()
