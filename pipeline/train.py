from copy import deepcopy
from time import perf_counter

import torch
from torch.nn import CrossEntropyLoss

from utils.io import save_checkpoint


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
                device, checkpoint_path=None, weight_decay=0.01,
                early_stopping_patience=10, lr_patience=3, min_delta=0.001):
    """Train up to num_epochs and restore the lowest-validation-loss weights.

    Halve the learning rate after lr_patience + 1 stalled epochs. Stop after
    early_stopping_patience epochs without a validation-loss drop of min_delta.
    Checkpoints still track the actual minimum loss, including smaller drops.
    History stays in memory; checkpoint file handling lives in utils.io.
    """
    model.to(device)
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=lr_patience,
        threshold=min_delta, threshold_mode="abs", min_lr=1e-6,
    )
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [],
               "learning_rate": [], "train_seconds": [], "validation_seconds": []}
    best_val_loss = patience_loss = float("inf")
    stalled_epochs = 0

    for epoch in range(1, num_epochs + 1):
        learning_rate_used = optimizer.param_groups[0]["lr"]
        # The .item() calls in these loops already wait for the GPU results.
        start = perf_counter()
        train_loss = train_one_epoch(model, train_dataloader, criterion, optimizer, device)
        train_seconds = perf_counter() - start

        start = perf_counter()
        val_loss, val_accuracy = validate(model, val_dataloader, criterion, device)
        validation_seconds = perf_counter() - start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["learning_rate"].append(learning_rate_used)
        history["train_seconds"].append(train_seconds)
        history["validation_seconds"].append(validation_seconds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = deepcopy(model.state_dict())
            history.update(best_epoch=epoch, best_val_loss=val_loss,
                           val_accuracy_at_best_loss=val_accuracy)
            if checkpoint_path is not None:
                save_checkpoint(model, checkpoint_path, epoch, val_loss, val_accuracy)

        # Small improvements can accumulate before resetting the patience counter.
        if val_loss < patience_loss - min_delta:
            patience_loss = val_loss
            stalled_epochs = 0
        else:
            stalled_epochs += 1
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{num_epochs}: train loss={train_loss:.4f}, "
              f"val loss={val_loss:.4f}, val accuracy={val_accuracy:.4f}, "
              f"lr={learning_rate_used:.2g}, "
              f"train={train_seconds:.2f}s, val={validation_seconds:.2f}s")
        next_lr = optimizer.param_groups[0]["lr"]
        if next_lr < learning_rate_used:
            print(f"Learning rate reduced to {next_lr:.2g}")
        if stalled_epochs >= early_stopping_patience:
            print(f"Early stopping: no meaningful validation-loss improvement "
                  f"for {early_stopping_patience} epochs.")
            break

    model.load_state_dict(best_weights)
    model.eval()
    print(f"Restored epoch {history['best_epoch']} "
          f"(lowest validation loss={history['best_val_loss']:.4f})")
    return history
