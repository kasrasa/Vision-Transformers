import torch
from torch import nn

def patchify(image_set, patch_size):
    """
    Flattens non-overlapping patches in raster order (left to right, top to bottom).

    Args:
        image_set (torch.Tensor): A tensor of shape (batch_size, channels, height, width).
        patch_size (int): The size of each square patch.

    Use image dimensions divisible by patch_size. Pixel values are left unchanged;
    convert images to floating point and normalize them before calling this function.
    """
    _, channels, height, width = image_set.shape
    embedding_dim = patch_size * patch_size * channels

    patches = []
    for row in range(0, height, patch_size):
        for col in range(0, width, patch_size):
            patch = image_set[:, :, row:row + patch_size, col:col + patch_size]
            patches.append(patch.flatten(start_dim=1))

    return {
        "patches": torch.stack(patches, dim=1),
        "embedding_dim": embedding_dim,
    }


class PatchEmbeddingModel(nn.Module):
    """Project patches and build the original ViT's input token sequence.

    embedding_dim is the flattened patch width (channels * patch_size ** 2).
    hidden_dim is the transformer's token width. Create this model once and reuse it.
    """

    def __init__(self, embedding_dim=768, hidden_dim=192, num_patches=196):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedder = nn.Linear(embedding_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.positional_encoding = PositionalEncoding(hidden_dim, num_patches + 1)

    def forward(self, x):
        """Map [B, N, patch_width] to patch embeddings and [B, N + 1, hidden_dim]."""
        embedded_patches = self.embedder(x)
        tokens = prepend_classification_token(embedded_patches, self.cls_token)
        tokens = self.positional_encoding(tokens)

        return {
            "embedded_patches": embedded_patches,
            "patches_with_cls_token": tokens,
            "hidden_dim": self.hidden_dim,
        }


class PositionalEncoding(nn.Module):
    """Learned absolute positions, including a position for the classification token."""

    def __init__(self, embedding_dim, num_tokens):
        super().__init__()
        self.pe = nn.Parameter(torch.empty(1, num_tokens, embedding_dim))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x):
        return x + self.pe


def prepend_classification_token(embedded_patches, cls_token):
    """
    Share the model's learned classification token across the current batch.

    Args:
        embedded_patches (torch.Tensor): A tensor of shape (batch_size, num_patches, hidden_dim).
        cls_token (torch.Tensor): A learned parameter of shape (1, 1, hidden_dim).
    """
    batch_size = embedded_patches.size(0)
    return torch.cat((cls_token.expand(batch_size, -1, -1), embedded_patches), dim=1)
