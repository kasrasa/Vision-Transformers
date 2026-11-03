import torch.nn as nn

from utils.patch_embedding import patchify


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
