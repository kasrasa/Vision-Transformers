from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from .checkpoints import load_checkpoint
from .predictions import prediction_log_probs


def evaluate_model(model, test_dataloader, device, checkpoint_path=None):
    """Score fixed weights; evaluation never selects or overwrites a checkpoint.

    evaluation_seconds covers dataset loading, transfers, forward passes, and
    metrics. Checkpoint loading is outside it; evaluation writes no files.
    """
    device = torch.device(device)
    model.to(device)
    if checkpoint_path is not None:
        load_checkpoint(model, checkpoint_path, device)
    model.eval()
    all_labels = []
    all_predictions = []
    total_loss = 0.0
    total_samples = 0
    start = perf_counter()

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = prediction_log_probs(outputs)
                total_loss += torch.nn.functional.nll_loss(outputs, labels, reduction="sum").item()
            else:
                # Preserve Part 1's original single-head loss calculation.
                total_loss += torch.nn.functional.cross_entropy(outputs, labels).item() * labels.size(0)
            total_samples += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())

    class_labels = list(range(outputs.size(1)))
    metrics = {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "f1_score": f1_score(all_labels, all_predictions, labels=class_labels, average="weighted", zero_division=0),
        "macro_f1_score": f1_score(all_labels, all_predictions, labels=class_labels, average="macro", zero_division=0),
        "precision": precision_score(all_labels, all_predictions, labels=class_labels, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_predictions, labels=class_labels, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_predictions, labels=class_labels),
        "loss": total_loss / total_samples,
    }
    metrics["evaluation_seconds"] = perf_counter() - start
    return metrics


def calculate_ece(model_prediction, true_labels, num_bins=10):
    """Top-label ECE for [N, classes] probabilities and [N] class indices.

    Convert logits with softmax before calling. Confidence is the largest class
    probability; accuracy is whether that class matches the target label.
    """
    model_prediction = np.asarray(model_prediction)
    true_labels = np.asarray(true_labels)
    confidences = model_prediction.max(axis=1)
    predicted_labels = model_prediction.argmax(axis=1)
    bins = np.linspace(0, 1, num_bins + 1)
    # Interior edges assign both confidence 0 and confidence 1 to valid bins.
    bin_indices = np.digitize(confidences, bins[1:-1])
    ece = 0.0
    for i in range(num_bins):
        bin_mask = bin_indices == i
        if np.sum(bin_mask) > 0:
            bin_accuracy = np.mean(true_labels[bin_mask] == predicted_labels[bin_mask])
            bin_confidence = np.mean(confidences[bin_mask])
            ece += np.abs(bin_accuracy - bin_confidence) * np.sum(bin_mask) / len(model_prediction)

    return ece

