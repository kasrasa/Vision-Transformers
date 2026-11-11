"""Shared supervised loss and predictions for one- and two-head classifiers."""

from math import log

import torch
from torch.nn import functional as F


def supervised_loss(outputs, targets):
    """Supervise each head with the same labels, weighting heads equally."""
    heads = outputs if isinstance(outputs, tuple) else (outputs,)
    return sum(F.cross_entropy(head, targets) for head in heads) / len(heads)


def prediction_log_probs(outputs):
    """Average head probabilities in log space, following the paper's fusion rule."""
    if isinstance(outputs, tuple):
        log_probs = torch.stack([F.log_softmax(head, dim=-1) for head in outputs])
        return torch.logsumexp(log_probs, dim=0) - log(len(outputs))
    return F.log_softmax(outputs, dim=-1)
