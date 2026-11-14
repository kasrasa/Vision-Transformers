import torch.nn as nn

from models.patch_embedding import PatchEmbeddingModel
from models.transformer_encoder import TransformerEncoderLayer
from models.mlp_head import MLPHead
from models.patch_embedding import patchify


class VisionTransformer(nn.Module):
    """Connect the existing components into an image-to-logits model.

    The supplied modules are registered once, so model.parameters(), to(),
    train(), and eval() include the patch embedder, encoder, final norm, and head.
    """

    def __init__(self, patch_embedder, transformer_encoder, encoder_norm, mlp_head, patch_size=16):
        super().__init__()
        self.patch_embedder = patch_embedder
        self.transformer_encoder = transformer_encoder
        self.encoder_norm = encoder_norm
        self.mlp_head = mlp_head
        self.patch_size = patch_size

    def forward(self, images):
        """Map [B, 3, H, W] images to [B, num_classes] logits."""
        patches = patchify(images, self.patch_size)["patches"]
        tokens = self.patch_embedder(patches)["patches_with_cls_token"]
        tokens = self.transformer_encoder(tokens)
        cls_representation = self.encoder_norm(tokens)[:, 0, :]
        return self.mlp_head(cls_representation)


class DistilledVisionTransformer(nn.Module):
    """Two special tokens with a shared encoder and independent linear heads.

    Both training and evaluation return (CLS logits, DIST logits). Keeping raw
    logits separate lets the training pipeline choose each supervision target
    and the evaluation pipeline average the two softmax probability vectors.
    The teacher and distillation loss do not belong to this model.
    """

    def __init__(self, patch_embedder, transformer_encoder, encoder_norm,
                 cls_head, dist_head, patch_size=16):
        super().__init__()
        self.patch_embedder = patch_embedder
        self.transformer_encoder = transformer_encoder
        self.encoder_norm = encoder_norm
        self.cls_head = cls_head
        self.dist_head = dist_head
        self.patch_size = patch_size

    def forward(self, images):
        """Map [B, 3, H, W] to two [B, num_classes] logit tensors."""
        patches = patchify(images, self.patch_size)["patches"]
        tokens = self.patch_embedder(patches)["patches_with_cls_token"]
        tokens = self.encoder_norm(self.transformer_encoder(tokens))
        return self.cls_head(tokens[:, 0, :]), self.dist_head(tokens[:, 1, :])


def build_vision_transformer(image_size=224, patch_size=16, hidden_dim=192,
                             num_heads=3, num_layers=1, mlp_ratio=4,
                             num_classes=10, dropout=0.1):
    """Assemble fresh components; choose divisible image/patch and width/head sizes.

    Each encoder block owns independent parameters. The final LayerNorm remains
    after the entire stack. Defaults preserve the small learning configuration.
    """
    model = VisionTransformer(
        patch_embedder=PatchEmbeddingModel(
            embedding_dim=3 * patch_size ** 2,
            hidden_dim=hidden_dim,
            num_patches=(image_size // patch_size) ** 2,
        ),
        transformer_encoder=nn.Sequential(*[
            TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * mlp_ratio, dropout)
            for _ in range(num_layers)
        ]),
        encoder_norm=nn.LayerNorm(hidden_dim),
        mlp_head=MLPHead(hidden_dim, hidden_dim * mlp_ratio, num_classes),
        patch_size=patch_size,
    )
    model.config = {
        "image_size": image_size, "patch_size": patch_size, "hidden_dim": hidden_dim,
        "num_heads": num_heads, "num_layers": num_layers, "mlp_ratio": mlp_ratio,
        "num_classes": num_classes, "dropout": dropout,
    }
    return model

def build_vision_transformer_single_classifier(image_size=224, patch_size=16, hidden_dim=192,
                             num_heads=3, num_layers=1, mlp_ratio=4,
                             num_classes=10, dropout=0.1):
    """Assemble fresh components; choose divisible image/patch and width/head sizes.

    Each encoder block owns independent parameters. The final LayerNorm remains
    after the entire stack. Defaults preserve the small learning configuration.
    """
    model = VisionTransformer(
        patch_embedder=PatchEmbeddingModel(
            embedding_dim=3 * patch_size ** 2,
            hidden_dim=hidden_dim,
            num_patches=(image_size // patch_size) ** 2,
        ),
        transformer_encoder=nn.Sequential(*[
            TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * mlp_ratio, dropout)
            for _ in range(num_layers)
        ]),
        encoder_norm=nn.LayerNorm(hidden_dim),
        mlp_head=nn.Linear(hidden_dim, num_classes),
        patch_size=patch_size,
    )
    model.config = {
        "image_size": image_size, "patch_size": patch_size, "hidden_dim": hidden_dim,
        "num_heads": num_heads, "num_layers": num_layers, "mlp_ratio": mlp_ratio,
        "num_classes": num_classes, "dropout": dropout,
    }
    return model

def build_distilled_vision_transformer(image_size=224, patch_size=16, hidden_dim=192,
                                      num_heads=4, num_layers=6, mlp_ratio=4,
                                      num_classes=10, dropout=0.1):
    """Build the small two-token student without changing the Part 1 builder.

    CLS and DIST each use one linear classifier. Encoder blocks, dropout, and
    initialization follow the existing components. Distillation is optional:
    the two heads can first be trained against the same ground-truth labels.
    """
    model = DistilledVisionTransformer(
        patch_embedder=PatchEmbeddingModel(
            embedding_dim=3 * patch_size ** 2,
            hidden_dim=hidden_dim,
            num_patches=(image_size // patch_size) ** 2,
            num_special_tokens=2,
        ),
        transformer_encoder=nn.Sequential(*[
            TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * mlp_ratio, dropout)
            for _ in range(num_layers)
        ]),
        encoder_norm=nn.LayerNorm(hidden_dim),
        cls_head=nn.Linear(hidden_dim, num_classes),
        dist_head=nn.Linear(hidden_dim, num_classes),
        patch_size=patch_size,
    )
    model.config = {
        "image_size": image_size, "patch_size": patch_size, "hidden_dim": hidden_dim,
        "num_heads": num_heads, "num_layers": num_layers, "mlp_ratio": mlp_ratio,
        "num_classes": num_classes, "dropout": dropout,
    }
    return model

