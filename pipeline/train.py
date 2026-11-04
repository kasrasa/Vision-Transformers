import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import CrossEntropyLoss

from utils.inference_utils import synchronize_device


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return running_loss / total_samples


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, correct_predictions / total_samples


def train_model(model, train_dataloader, val_dataloader, num_epochs, learning_rate,
                device, checkpoint_path=None, experiment_config=None, weight_decay=0.01):
    """Fit for a positive epoch budget and restore the best validation weights.

    Highest validation accuracy wins; lower validation loss breaks ties. Epoch
    timings include loading and device transfers but exclude checkpoint writes.
    total_seconds includes the full fit loop, saving, and best-weight restoration.
    Notebook experiments provide distinct checkpoint paths; callers omitting a
    path still receive the best model in memory. Checkpoints are for evaluation,
    not automatic optimizer resumption.
    """
    device = torch.device(device)
    model.to(device)
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [],
               "train_seconds": [], "validation_seconds": [], "checkpoint_seconds": []}
    experiment_config = {} if experiment_config is None else experiment_config
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    synchronize_device(device)
    fit_start = perf_counter()

    for epoch in range(num_epochs):
        synchronize_device(device)
        start = perf_counter()
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        synchronize_device(device)
        train_seconds = perf_counter() - start

        start = perf_counter()
        val_loss, val_accuracy = validate(model, val_dataloader, criterion, device)
        synchronize_device(device)
        validation_seconds = perf_counter() - start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["train_seconds"].append(train_seconds)
        history["validation_seconds"].append(validation_seconds)

        checkpoint_seconds = 0.0
        improved = (val_accuracy > best_val_accuracy or
                    (val_accuracy == best_val_accuracy and val_loss < best_val_loss))
        if improved:
            start = perf_counter()
            best_epoch = epoch + 1
            best_val_accuracy, best_val_loss = val_accuracy, val_loss
            # Clone parameters AND buffers: state_dict() alone references live state.
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
            if checkpoint_path is not None:
                torch.save({
                    "model_state_dict": best_state,
                    "epoch": best_epoch,
                    "validation_accuracy": best_val_accuracy,
                    "validation_loss": best_val_loss,
                    "model_name": model.__class__.__name__,
                    "model_config": getattr(model, "config", {}),
                    "experiment_config": experiment_config,
                    "training_config": {"epochs": num_epochs, "learning_rate": learning_rate,
                                        "weight_decay": weight_decay, "optimizer": "AdamW"},
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "torch_version": str(torch.__version__),
                }, checkpoint_path)
            checkpoint_seconds = perf_counter() - start
        history["checkpoint_seconds"].append(checkpoint_seconds)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, "
              f"Train: {train_seconds:.3f}s, Validation: {validation_seconds:.3f}s")

    model.load_state_dict(best_state)
    model.eval()
    synchronize_device(device)
    history.update({
        "best_epoch": best_epoch, "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss, "total_seconds": perf_counter() - fit_start,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "device": str(device), "cpu_threads": torch.get_num_threads(),
        "torch_version": str(torch.__version__),
        "experiment_config": experiment_config,
        "training_config": {"epochs": num_epochs, "learning_rate": learning_rate,
                            "weight_decay": weight_decay, "optimizer": "AdamW"},
    })
    if checkpoint_path is not None:
        history_path = checkpoint_path.with_suffix(".history.json")
        history["history_path"] = str(history_path)
        history_path.write_text(json.dumps(history, indent=2) + "\n")
    print(f"Restored best validation weights from epoch {best_epoch} "
          f"(accuracy={best_val_accuracy:.4f}, loss={best_val_loss:.4f})")
    return history
