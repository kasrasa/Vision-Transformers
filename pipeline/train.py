from copy import deepcopy
from math import cos, pi
from time import perf_counter

import torch
from torch.nn import CrossEntropyLoss

from utils.io import save_checkpoint
from utils.scheduler import linear_scheduler, cosine_decay_scheduler

def train_one_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
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
        if scheduler is not None:
            scheduler.step()
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
                scheduler = "linear", warmup_steps=None, min_delta=0.001):
    """Train with warmup/cosine decay and restore the best-validation-accuracy weights.

    learning_rate is the peak LR. Warmup defaults to 10% of planned updates,
    adapting ViT's ImageNet pretraining schedule to this smaller experiment.
    Set warmup_steps explicitly for a different budget, shorter than total_steps.
    LR changes after every optimizer update; history records each epoch's start LR.
    Optional early stopping counts ten stalled epochs after warmup, using accuracy
    gains greater than min_delta. Disabling it allows the full cosine schedule.
    """
    early_stopping_patience = 10

    model.to(device)
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if scheduler == "linear":
        scheduler = linear_scheduler(optimizer, lr_patience=3, threshold=min_delta)
    elif scheduler == "cosine_decay":
        steps_per_epoch = len(train_dataloader)
        total_steps = num_epochs * steps_per_epoch 
        warmup_steps = int(0.1 * total_steps) if warmup_steps is None else warmup_steps
        scheduler = cosine_decay_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [],
               "learning_rate": [], "train_seconds": [], "validation_seconds": []}
    best_val_accuracy = patience_accuracy = -1.0
    stalled_epochs = 0

    for epoch in range(1, num_epochs + 1):
        learning_rate_used = optimizer.param_groups[0]["lr"]
        # The .item() calls in these loops already wait for the GPU results.
        start = perf_counter()
        train_loss = train_one_epoch(
            model, train_dataloader, criterion, optimizer, device, scheduler,
        )
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

        # Warmup does not consume patience; small accuracy gains can accumulate.
        if scheduler and epoch * steps_per_epoch >= warmup_steps:
            if val_accuracy > patience_accuracy + min_delta:
                patience_accuracy = val_accuracy
                stalled_epochs = 0
            else:
                stalled_epochs += 1

        print(f"Epoch {epoch}/{num_epochs}: train loss={train_loss:.4f}, "
              f"val loss={val_loss:.4f}, val accuracy={val_accuracy:.4f}, "
              f"lr_start={learning_rate_used:.2g}, "
              f"train={train_seconds:.2f}s, val={validation_seconds:.2f}s")
        if scheduler and stalled_epochs >= early_stopping_patience:
            print(f"Early stopping: no meaningful validation-accuracy improvement "
                  f"for {early_stopping_patience} epochs after warmup.")
            break

    model.load_state_dict(best_weights)
    model.eval()
    print(f"Restored epoch {history['best_epoch']} "
          f"(highest validation accuracy={history['best_val_accuracy']:.4f})")
    return history

