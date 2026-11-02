import matplotlib.pyplot as plt
import numpy as np


def display_patches(patches, num_patches_to_display = 5, figsize=(15, 3)):
    """
    Displays a specified number of image patches randomly in the image.

    Args:
        patches (torch.Tensor): A tensor containing batches of image patches.
        num_patches_to_display (int): Number of patches to display.
        figsize (tuple): Size of the figure.
    """
    # Randomly select batches and patches to display
    batch_size = num_patches_to_display if patches.size(0) > 1 else 1
    batch_indices = np.random.choice(patches.size(0), batch_size, replace=False)
    patch_indices = np.random.randint(0, patches.size(1), num_patches_to_display)
    selected_patches = patches[batch_indices][:, patch_indices]

    # Create a figure to display the patches
    plt.figure(figsize=figsize)
    for i in range(num_patches_to_display):
        plt.subplot(1, num_patches_to_display, i + 1)
        plt.imshow(selected_patches[i].permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
    plt.show()