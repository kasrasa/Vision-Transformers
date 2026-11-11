import torch
from math import cos, pi

def linear_scheduler(optimizer, lr_patience, threshold, mode="min", min_lr=1e-6):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=0.5, patience=lr_patience,
            threshold=threshold, threshold_mode="abs", min_lr=min_lr,
        )

    return scheduler

def warmup_cosine_factor(step, warmup_steps, total_steps):
    """ViT's linear warmup followed by cosine decay to zero, per update."""
    progress = min(max((step - warmup_steps) / (total_steps - warmup_steps), 0.0), 1.0)
    factor = 0.5 * (1.0 + cos(pi * progress))
    if warmup_steps:
        factor *= min(1.0, step / warmup_steps)
    return factor

def cosine_decay_scheduler(optimizer, warmup_steps, total_steps):
    scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: warmup_cosine_factor(step, warmup_steps, total_steps),
        )

    return scheduler