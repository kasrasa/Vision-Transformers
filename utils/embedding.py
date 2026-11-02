import torch
from torch import nn

def patchify(image_set, patch_size):
    """
    Divides batches of images into non-overlapping patches and flattens them.

    Args:
        image_set (torch.Tensor): A tensor of shape (batch_size, channels, height, width).
        patch_size (int): The size of each square patch.
    """
    batch_size, channels, height, width = image_set.shape

    assert height % patch_size == 0 and width % patch_size == 0, \
        "Image dimensions must be divisible by the patch size."

    embedding_dim = patch_size * patch_size * channels
    num_patches = (height // patch_size) * (width // patch_size)

    patches = torch.zeros((batch_size, num_patches, embedding_dim), dtype=torch.float32)

    for k in range(batch_size):
        for i in range(0, width, patch_size):
            for j in range(0, height, patch_size):
                box = (i, j, i + patch_size, j + patch_size)
                patch = image_set[k, :, j:j+patch_size, i:i+patch_size]
                patch_tensor = patch.float() / 255.0
                patch_tensor_flattened = patch_tensor.reshape(-1)
                patches[k, (i // patch_size) * (height // patch_size) + (j // patch_size), :] = patch_tensor_flattened

    output = {
        "patches": patches,
        "embedding_dim": embedding_dim
    }
    return output


class PatchEmbeddingModel(torch.nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim = 192):
        super(PatchEmbeddingModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedder = nn.Linear(self.embedding_dim, self.hidden_dim)

    def forward(self, x):
        embedded_patches = self.embedder(x)
        return embedded_patches, self.hidden_dim

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
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

def prepend_classification_token(embedded_patches, hidden_dim):
    """
    Prepends a classification token to the embedded patches.

    Args:
        embedded_patches (torch.Tensor): A tensor of shape (batch_size, num_patches, hidden_dim).
        hidden_dim (int): The dimension of the hidden layer.
    """
    batch_size = embedded_patches.size(0)
    cls_token = torch.zeros((batch_size, 1, hidden_dim), dtype=torch.float32)
    embedded_patches_with_cls_token = torch.cat((cls_token, embedded_patches), dim=1)

    return embedded_patches_with_cls_token

def prepare_patch_embeddings(patches, embedding_dim):
    """
    embeds the patches and adds positional encoding to them.

    Args:
        patches (torch.Tensor): A tensor of shape (batch_size, num_patches, embedding_dim).
        positional_encoding (torch.Tensor): A tensor of shape (1, num_patches, embedding_dim) representing the positional encoding.
    """
    patch_embedding_model = PatchEmbeddingModel(embedding_dim=embedding_dim)
    embedded_patches, hidden_dim = patch_embedding_model(patches)

    positional_encoding = PositionalEncoding(embedding_dim=hidden_dim)
    embedded_patches_with_positional_encoding = positional_encoding(embedded_patches)

    patches_with_cls_token = prepend_classification_token(embedded_patches_with_positional_encoding, hidden_dim)

    output = {
        "embedded_patches": embedded_patches_with_positional_encoding,
        "patches_with_cls_token": patches_with_cls_token,
        "hidden_dim": hidden_dim
    }
    return output