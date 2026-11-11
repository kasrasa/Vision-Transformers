"""Learning-rate schedules stepped after each optimizer update."""

from math import cos, pi

from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau


def linear_scheduler_part1(optimizer, lr_patience=3, threshold=0.001, mode="min", min_lr=1e-6):
    """Keep Part 1's original function name and plateau-reduction settings."""
    return ReduceLROnPlateau(
        optimizer, mode=mode, factor=0.5, patience=lr_patience,
        threshold=threshold, threshold_mode="abs", min_lr=min_lr,
    )

def _warmup_decay_scheduler(optimizer, warmup_steps, total_steps, decay,
                            warmup_lr, min_lr):
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("Require 0 <= warmup_steps < total_steps")

    peak_lr = optimizer.param_groups[0]["lr"]
    if peak_lr <= 0 or not 0 <= min_lr <= peak_lr or not 0 <= warmup_lr <= peak_lr:
        raise ValueError("Require a positive peak rate and warmup/minimum rates between zero and peak")

    def factor(step):
        # LambdaLR applies step 0 before the first optimizer update.
        if step < warmup_steps:
            progress = step / max(1, warmup_steps - 1)
            return (warmup_lr + (peak_lr - warmup_lr) * progress) / peak_lr
        decay_steps = total_steps - warmup_steps
        progress = min((step - warmup_steps) / max(1, decay_steps - 1), 1.0)
        return (min_lr + (peak_lr - min_lr) * decay(progress)) / peak_lr

    return LambdaLR(optimizer, factor)


def linear_scheduler(optimizer, warmup_steps, total_steps, warmup_lr=1e-6, min_lr=0.0):
    """Linear warmup, then linear decay; total_steps is the full update budget."""
    return _warmup_decay_scheduler(
        optimizer, warmup_steps, total_steps, lambda progress: 1.0 - progress,
        warmup_lr, min_lr,
    )


def cosine_decay_scheduler(optimizer, warmup_steps, total_steps, warmup_lr=1e-6, min_lr=0.0):
    """Linear warmup, then a single cosine decay without restarts."""
    return _warmup_decay_scheduler(
        optimizer, warmup_steps, total_steps,
        lambda progress: 0.5 * (1.0 + cos(pi * progress)), warmup_lr, min_lr,
    )
