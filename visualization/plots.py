import matplotlib.pyplot as plt
import numpy as np


def display_patches(patches, patch_size, num_patches_to_display=5, figsize=(15, 3)):
    """
    Display random patches from the flattened output of patchify.

    Args:
        patches (torch.Tensor): A tensor of shape (batch_size, num_patches, channels * patch_size ** 2).
        patch_size (int): The side length used when extracting patches.
        num_patches_to_display (int): Number of patches to display.
        figsize (tuple): Size of the figure.
    """
    channels = patches.size(-1) // (patch_size ** 2)
    batch_indices = np.random.randint(0, patches.size(0), num_patches_to_display)
    patch_indices = np.random.randint(0, patches.size(1), num_patches_to_display)

    _, axes = plt.subplots(1, num_patches_to_display, figsize=figsize, squeeze=False)
    for axis, batch_index, patch_index in zip(axes[0], batch_indices, patch_indices):
        patch = patches[batch_index, patch_index].reshape(channels, patch_size, patch_size)
        axis.imshow(patch.permute(1, 2, 0).detach().cpu().numpy())
        axis.set_title(f"Image {batch_index}, patch {patch_index}")
        axis.axis('off')
    plt.show()


def plot_history(history):
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    val_accuracy = history["val_accuracy"]
    epochs = range(1, len(train_loss) + 1)

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_loss, label="Training loss")
    axes[0].plot(epochs, val_loss, label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and validation loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, val_accuracy, label="Validation accuracy", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
