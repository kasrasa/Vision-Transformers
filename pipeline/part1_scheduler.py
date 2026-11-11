"""Part 1's original validation-driven schedule; independent of Part 2."""

from torch.optim.lr_scheduler import ReduceLROnPlateau


def linear_scheduler(optimizer, lr_patience=3, threshold=0.001, mode="min", min_lr=1e-6):
    """Keep Part 1's original function name and plateau-reduction settings."""
    return ReduceLROnPlateau(
        optimizer, mode=mode, factor=0.5, patience=lr_patience,
        threshold=threshold, threshold_mode="abs", min_lr=min_lr,
    )
