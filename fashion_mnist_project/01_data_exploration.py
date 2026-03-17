import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

np.random.seed(42)
torch.manual_seed(42)

from utils import CLASS_NAMES, load_data

def main():
    os.makedirs('outputs/figures', exist_ok=True)

    # Load full training set for exploration
    transform = transforms.ToTensor()
    full_train = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform)

    # Extract all images and labels
    train_loader = torch.utils.data.DataLoader(full_train, batch_size=len(full_train))
    all_images, all_labels = next(iter(train_loader))
    all_images_np = all_images.numpy()

    # Also load the split data for size reporting
    train_ds, val_ds, test_ds = load_data()

    # Print dataset info
    print(f"Training set size: {len(train_ds)}")
    print(f"Validation set size: {len(val_ds)}")
    print(f"Test set size: {len(test_ds)}")
    print(f"Full training set (before split): {len(full_train)}")
    print(f"Image shape: {all_images_np.shape[1:]}")
    print(f"Min pixel value: {all_images_np.min():.4f}")
    print(f"Max pixel value: {all_images_np.max():.4f}")
    print(f"Mean pixel value: {all_images_np.mean():.4f}")

    # Check class balance
    unique, counts = np.unique(all_labels.numpy(), return_counts=True)
    print("\nClass distribution:")
    for cls_id, count in zip(unique, counts):
        print(f"  {CLASS_NAMES[cls_id]}: {count}")
    if counts.max() - counts.min() == 0:
        print("Classes are perfectly balanced.")
    else:
        print(f"Classes are approximately balanced (range: {counts.min()}-{counts.max()}).")

    # Figure 1: Sample grid (5 examples per class, 10 classes)
    fig, axes = plt.subplots(5, 10, figsize=(15, 8))
    for cls_id in range(10):
        cls_indices = (all_labels == cls_id).nonzero(as_tuple=True)[0]
        selected = cls_indices[:5]
        for row in range(5):
            ax = axes[row, cls_id]
            ax.imshow(all_images_np[selected[row], 0], cmap='gray')
            ax.axis('off')
            if row == 0:
                ax.set_title(CLASS_NAMES[cls_id], fontsize=8)
    plt.suptitle('Sample Images per Class', fontsize=14)
    plt.savefig('outputs/figures/sample_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved: outputs/figures/sample_grid.png")

    # Figure 2: Class distribution bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(10), counts, tick_label=CLASS_NAMES)
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution in Training Set')
    plt.xticks(rotation=45, ha='right')
    plt.savefig('outputs/figures/class_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/figures/class_distribution.png")

    # Figure 3: Mean image per class (2x5 grid)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for cls_id in range(10):
        row, col = cls_id // 5, cls_id % 5
        cls_mask = all_labels.numpy() == cls_id
        mean_img = all_images_np[cls_mask, 0].mean(axis=0)
        ax = axes[row, col]
        ax.imshow(mean_img, cmap='gray')
        ax.set_title(CLASS_NAMES[cls_id], fontsize=9)
        ax.axis('off')
    plt.suptitle('Mean Image per Class', fontsize=14)
    plt.savefig('outputs/figures/mean_per_class.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/figures/mean_per_class.png")

    # Figure 4: Pixel intensity histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_images_np.flatten(), bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Pixel Intensity')
    ax.set_ylabel('Frequency')
    ax.set_title('Pixel Intensity Distribution')
    plt.savefig('outputs/figures/pixel_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: outputs/figures/pixel_histogram.png")

    print("\nData exploration complete!")


if __name__ == '__main__':
    main()
