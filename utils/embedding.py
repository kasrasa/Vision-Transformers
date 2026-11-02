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
        tokens = torch.cat([self.cls_token.expand(x.size(0), -1, -1), embedded_patches], dim=1)
        tokens = self.positional_encoding(tokens)

        return {
            "embedded_patches": embedded_patches,
            "patches_with_cls_token": tokens,
            "cls_token": self.cls_token,
            "positional_encoding": self.positional_encoding.pe,
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


class SinusoidalPositionalEncoding(nn.Module):
    """Original fixed sine/cosine variant, kept for a later comparison."""

    def __init__(self, embedding_dim, max_len=5000):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Create a matrix of shape (max_len, embedding_dim) to hold the positional encodings
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # each token's position
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): A tensor of shape (batch_size, num_patches, embedding_dim).
        """
        x = x + self.pe[:, :x.size(1), :]
        return x
