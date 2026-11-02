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
