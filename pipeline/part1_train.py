"""Preserve Part 1's training policy separately from the Part 2 experiments."""

from copy import deepcopy
from time import perf_counter

import torch
from torch.nn import CrossEntropyLoss

from .checkpoints import save_checkpoint
from .scheduler import linear_scheduler_part1


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
    return running_loss / total_samples


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = correct_predictions = total_samples = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item() * images.size(0)
            correct_predictions += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)
    return total_loss / total_samples, correct_predictions / total_samples


def train_model(model, train_dataloader, val_dataloader, num_epochs, learning_rate,
                device, checkpoint_path=None, weight_decay=0.01,
                early_stopping_patience=10, lr_patience=3, min_delta=0.001):
    """Use Part 1's original plateau schedule and accuracy-based early stopping.

    num_epochs is a maximum, not a fixed training duration. The learning rate
    starts at learning_rate without warmup and halves after four stalled
    validation-loss epochs by default. Checkpoints keep the highest validation
    accuracy (earlier epoch on ties); stopping counts accuracy gains > min_delta.
    These are the settings in the user's Part 1 code before the Part 2 cleanup.
    """
    model.to(device)
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = linear_scheduler_part1(optimizer, lr_patience=lr_patience, threshold=min_delta)
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [],
               "learning_rate": [], "train_seconds": [], "validation_seconds": []}
    best_val_accuracy = patience_accuracy = -1.0
    stalled_epochs = 0

    for epoch in range(1, num_epochs + 1):
        learning_rate_used = optimizer.param_groups[0]["lr"]
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

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_weights = deepcopy(model.state_dict())
            history.update(best_epoch=epoch, best_val_accuracy=val_accuracy,
                           val_loss_at_best_accuracy=val_loss)
            if checkpoint_path is not None:
                save_checkpoint(model, checkpoint_path, epoch, val_loss, val_accuracy)

        if val_accuracy > patience_accuracy + min_delta:
            patience_accuracy = val_accuracy
            stalled_epochs = 0
        else:
            stalled_epochs += 1
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{num_epochs}: train loss={train_loss:.4f}, "
              f"val loss={val_loss:.4f}, val accuracy={val_accuracy:.4f}, "
              f"lr={learning_rate_used:.2g}, train={train_seconds:.2f}s, val={validation_seconds:.2f}s")
        if stalled_epochs >= early_stopping_patience:
            print("Early stopping: validation accuracy stalled.")
            break

    model.load_state_dict(best_weights)
    model.eval()
    print(f"Restored epoch {history['best_epoch']} "
          f"(highest validation accuracy={history['best_val_accuracy']:.4f})")
    return history
