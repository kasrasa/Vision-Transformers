"""Shared supervised loss and predictions for one- and two-head classifiers."""

from math import isfinite, log

import torch
from torch.nn import functional as F
import torch.nn as nn


def supervised_loss(images, outputs, targets):
    """Supervise each head equally; images are unused in this criterion."""
    heads = outputs if isinstance(outputs, tuple) else (outputs,)
    return sum(F.cross_entropy(head, targets) for head in heads) / len(heads)


def prediction_log_probs(outputs):
    """Average head probabilities in log space, following the paper's fusion rule."""
    if isinstance(outputs, tuple):
        log_probs = torch.stack([F.log_softmax(head, dim=-1) for head in outputs])
        return torch.logsumexp(log_probs, dim=0) - log(len(outputs))
    return F.log_softmax(outputs, dim=-1)


class DistillationLoss(nn.Module):
    """CLS label loss plus DIST teacher loss, using DeiT's released KL scaling.

    Ground-truth targets may already be mixed and smoothed. Teacher targets
    receive no extra smoothing, and the teacher stays frozen in evaluation mode.
    """

    def __init__(self, teacher, mode="hard", alpha=0.5, temperature=1.0):
        super().__init__()
        if mode not in ("hard", "soft"):
            raise ValueError("mode must be 'hard' or 'soft'")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if not isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        self.teacher = teacher.requires_grad_(False).eval()
        self.mode = mode
        self.alpha = float(alpha)
        self.temperature = float(temperature)

    @property
    def config(self):
        return {
            "mode": self.mode,
            "alpha": self.alpha,
            "temperature": self.temperature,
            "soft_normalization": "sum/numel" if self.mode == "soft" else None,
        }

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(self, images, outputs, targets):
        cls_logits, dist_logits = outputs
        label_loss = F.cross_entropy(cls_logits, targets)

        with torch.no_grad():
            teacher_logits = self.teacher(images)

        if self.mode == "hard":
            teacher_labels = teacher_logits.argmax(dim=1)
            teacher_loss = F.cross_entropy(dist_logits, teacher_labels)
        else:
            teacher_loss = F.kl_div(
                F.log_softmax(dist_logits / self.temperature, dim=1),
                F.log_softmax(teacher_logits / self.temperature, dim=1),
                log_target=True,
                reduction="sum",
            )
            teacher_loss *= self.temperature**2 / dist_logits.numel()

        return (
            (1 - self.alpha) * label_loss
            + self.alpha * teacher_loss
        )
