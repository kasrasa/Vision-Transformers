"""Small setup and artifact helpers shared by the experiment notebooks."""

import json
import random
from pathlib import Path

import numpy as np
import torch


def reset_seed(seed):
    """Reset the random generators used by the existing experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, payload):
    """Save experiment metadata, including NumPy arrays and scalar values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=lambda value: value.tolist()) + "\n"
    )
