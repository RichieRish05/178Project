import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def load_data():
    """Load Fashion-MNIST and return train/val/test splits."""
    transform = transforms.ToTensor()

    full_train = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform)

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train, [48000, 12000], generator=generator
    )

    return train_dataset, val_dataset, test_dataset


def get_flat_arrays(train_dataset, val_dataset, test_dataset):
    """Flatten image tensors to numpy arrays for sklearn models."""
    def extract(dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
        images, labels = next(iter(loader))
        return images.view(images.size(0), -1).numpy(), labels.numpy()

    X_train, y_train = extract(train_dataset)
    X_val, y_val = extract(val_dataset)
    X_test, y_test = extract(test_dataset)
    return X_train, y_train, X_val, y_val, X_test, y_test


def apply_pca(X_train, X_val, X_test, n_components=50):
    """Fit PCA on train, transform all three splits."""
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_val_pca, X_test_pca


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """Save a labeled confusion matrix PNG using seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_results(model_name, val_acc, test_acc, train_time,
                 filepath='outputs/results/results.csv'):
    """Append a row to the results CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    row = pd.DataFrame([{
        'model': model_name,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'train_time_sec': train_time
    }])
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df = pd.concat([df, row], ignore_index=True)
    else:
        df = row
    df.to_csv(filepath, index=False)


if __name__ == '__main__':
    print("Testing utils imports...")
    train, val, test = load_data()
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print("All imports OK.")
