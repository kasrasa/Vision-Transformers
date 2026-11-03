from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import torch
import numpy as np

def evaluate_model(model, test_dataloader, device):
    """Report weighted multiclass metrics for an image-to-logits model."""
    model.to(device)
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_dataloader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted_labels = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    class_labels = list(range(outputs.size(1)))
    f1 = f1_score(all_labels, all_predictions, labels=class_labels, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_predictions, labels=class_labels, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, labels=class_labels, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_predictions, labels=class_labels)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": conf_matrix
    }

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
