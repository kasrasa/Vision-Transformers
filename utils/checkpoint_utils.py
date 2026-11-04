import torch


def load_best_checkpoint(model, checkpoint_path, device):
    """Restore weights selected on validation; return their saved metadata."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return checkpoint
