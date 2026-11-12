from copy import deepcopy
from time import perf_counter

import torch
from torch.nn import functional as F

from .checkpoints import save_checkpoint
from .losses import prediction_log_probs, supervised_loss
from .scheduler import cosine_decay_scheduler, linear_scheduler


def train_one_epoch(model, dataloader, optimizer, device, scheduler, criterion,
                    batch_transform=None):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        if batch_transform is not None:
            images, labels = batch_transform(images, labels)

        optimizer.zero_grad()
        if isinstance(criterion, classmethod):
            loss = criterion(images, model(images), labels)
        else:
            loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return running_loss / total_samples


def validate(model, dataloader, device):
    """Score the inference prediction, including probability fusion for two heads."""
    model.eval()
    total_loss = correct_predictions = total_samples = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            log_probs = prediction_log_probs(model(images))
            total_loss += F.nll_loss(log_probs, labels, reduction="sum").item()
            correct_predictions += (log_probs.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)
    return total_loss / total_samples, correct_predictions / total_samples


def train_model(model, train_dataloader, val_dataloader, num_epochs, learning_rate,
                device, checkpoint_path=None, training_loss_fn=supervised_loss, weight_decay=0.01,
                scheduler="linear", warmup_steps=None, min_delta=0.001,
                early_stopping_patience=10, warmup_lr=1e-6, min_lr=0.0,
                batch_transform=None):
    """Train with per-update warmup/decay and restore highest-validation-accuracy weights.

    Warmup defaults to 10% of updates. Pass early_stopping_patience=None for a
    fixed budget. For two heads, both learn the same labels; no teacher is used.
    Checkpoint ties keep the earlier epoch. min_delta affects stopping only.
    batch_transform receives a batch of images and labels and returns both;
    it is applied once during training, never during validation.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    steps_per_epoch = len(train_dataloader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(0.1 * total_steps) if warmup_steps is None else warmup_steps
    schedulers = {"linear": linear_scheduler, "cosine_decay": cosine_decay_scheduler}
    if scheduler not in schedulers:
        raise ValueError(f"Unknown scheduler {scheduler!r}; choose 'linear' or 'cosine_decay'")
    lr_scheduler = schedulers[scheduler](
        optimizer, warmup_steps, total_steps, warmup_lr=warmup_lr, min_lr=min_lr,
    )
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [],
               "learning_rate": [], "train_seconds": [], "validation_seconds": []}
    best_val_accuracy = patience_accuracy = -1.0
    stalled_epochs = 0

    for epoch in range(1, num_epochs + 1):
        learning_rate_used = optimizer.param_groups[0]["lr"]
        start = perf_counter()
        train_loss = train_one_epoch(
            model, train_dataloader, optimizer, device, lr_scheduler,
            criterion=training_loss_fn, batch_transform=batch_transform
        )
        train_seconds = perf_counter() - start

        start = perf_counter()
        val_loss, val_accuracy = validate(model, val_dataloader, device)
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

        if epoch * steps_per_epoch >= warmup_steps:
            if val_accuracy > patience_accuracy + min_delta:
                patience_accuracy = val_accuracy
                stalled_epochs = 0
            else:
                stalled_epochs += 1

        print(f"Epoch {epoch}/{num_epochs}: train loss={train_loss:.4f}, "
              f"val loss={val_loss:.4f}, val accuracy={val_accuracy:.4f}, "
              f"lr_start={learning_rate_used:.2g}, "
              f"train={train_seconds:.2f}s, val={validation_seconds:.2f}s")
        if early_stopping_patience is not None and stalled_epochs >= early_stopping_patience:
            print("Early stopping: validation accuracy stalled after warmup.")
            break

    model.load_state_dict(best_weights)
    model.eval()
    print(f"Restored epoch {history['best_epoch']} "
          f"(highest validation accuracy={history['best_val_accuracy']:.4f})")
    return history
