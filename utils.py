import math
import cv2 as cv
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import numpy as np
from collections import Counter
from typing import Union, List, Tuple
import pandas as pd


def resize_image(img, size=(224, 224)):
    """
    Process image for model input with proper color handling.

    :param img: Input image (BGR format from OpenCV)
    :param size: Target size tuple
    :return: Processed tensor
    """
    if img is None:
        raise ValueError("Input image is None")

    if len(img.shape) != 3:
        raise ValueError(f"Expected 3 channels, got shape {img.shape}")

    # Convert BGR to RGB after all other processing
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        # TODO: Calculate these values from your face dataset
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    try:
        img_tensor = transform(img)
    except Exception as e:
        raise RuntimeError(f"Failed to transform image: {str(e)}")

    return img_tensor

def analyze_label_distribution(data) -> List[Tuple[int, int]]:
    """
    Analyzes the distribution of labels in a dataset with one-hot encoded vectors.

    :param data: FaceDataset object containing one-hot encoded labels
    :return: Sorted list of tuples (label_index, count), sorted by count in descending order
    """
    # convert the labels to indices
    label_indices = [torch.argmax(label).item() for label in data]

    # Count occurrences of each label
    counter = Counter(label_indices)

    # check if the sum of the values in the counter is equal to the length of the labels
    assert sum(counter.values()) == len(label_indices), "The sum of the values in the counter is not equal to the length of the labels"

    # Convert to list of tuples and sort by count in descending order
    distribution = [(int(label), int(count)) for label, count in counter.items()]
    distribution.sort(key=lambda x: x[1], reverse=True)

    return distribution


def print_distribution_summary(distribution: List[Tuple[int, int]], class_names: dict = None):
    """
    Prints a formatted summary of the label distribution.

    :param distribution: List of tuples (label_index, count) from analyze_label_distribution
    :param class_names: Optional dictionary mapping label indices to class names
    """
    total_samples = sum(count for _, count in distribution)

    # Create a DataFrame for nice formatting
    df_data = {
        'Label Index': [d[0] for d in distribution],
        'Count': [d[1] for d in distribution],
        'Percentage': [f"{(d[1] / total_samples) * 100:.2f}%" for d in distribution]
    }

    if class_names:
        df_data['Class Name'] = [class_names[idx] for idx, _ in distribution]

    df = pd.DataFrame(df_data)
    print("\nLabel Distribution Summary:")
    print(f"Total samples: {total_samples}")
    print("\nDistribution by class:")
    print(df.to_string(index=False))


def print_batch_labels(outputs, labels, class_names: dict):
    """
    Print the predicted and true labels for a batch of images.

    :param outputs: Tensor of predicted labels
    :param labels: Tensor of true labels
    :param class_names: Dictionary mapping label indices to class names
    """
    _, predicted = outputs.max(1)
    _, labels = labels.max(1)
    for i in range(len(predicted)):
        print(f"Predicted: {class_names[predicted[i]]}, True: {class_names[labels[i]]}")


def calculate_corrects(outputs, labels):
    """
    Calculate number of correct predictions with one-hot encoded labels.

    :param outputs: Model output tensor (N, num_classes)
    :param labels: One-hot encoded label tensor (N, num_classes)
    :return: Number of correct predictions
    """
    if outputs.size() != labels.size():
        raise ValueError(f"Output shape {outputs.size()} doesn't match labels shape {labels.size()}")

    if outputs.dim() != 2 or labels.dim() != 2:
        raise ValueError("Expected 2D tensors for outputs and labels")

    predicted = torch.argmax(outputs, dim=1)
    true_classes = torch.argmax(labels, dim=1)

    if len(predicted) == 0:
        return 0

    return (predicted == true_classes).sum().item()


def calculate_class_weights(data, num_classes: int, method='inverse'):
    """
    Calculate class weights based on dataset distribution.
    :param data: Dataset containing the samples
    :param num_classes: Total number of classes
    :param method: Weight calculation method ('inverse' or 'balanced')
                - 'inverse': 1/frequency
                - 'balanced': 1/(frequency * num_samples)

    :return: Class weights tensor
    """
    # Get label distribution
    distribution = analyze_label_distribution(data)

    # Create a weights tensor initialized with zeros
    weights = torch.zeros(num_classes)

    # Fill in weights based on distribution
    for label_idx, count in distribution:
        if method == 'inverse':
            # Inverse frequency weighting
            weights[label_idx] = 1.0 / count if count > 0 else 0.0
        elif method == 'balanced':
            # Balanced weighting
            weights[label_idx] = 1.0 / (count * num_classes) if count > 0 else 0.0

    # Normalize weights so they sum to num_classes
    weights *= (num_classes / weights.sum())

    return weights


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function
    between the initial lr set in the optimizer to 0, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.
    :param optimizer: the optimizer used for the training
    :param num_warmup_steps: the number of warmup steps
    :param num_training_steps: the total number of training steps
    :param num_cycles: the number of waves in the cosine schedule
    """

    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_weighted_loss_func(dataset, classes, device, confidence_penalty=0.1):
    """
    Create a weighted loss function that works with one-hot encoded labels and includes
    a confidence penalty to prevent overconfident predictions.

    Args:
        dataset: Dataset object containing the samples
        classes: List of class names
        device: Device to use for calculations
        confidence_penalty: Weight for the confidence penalty term (default: 0.1)
    """
    # Count instances of each class
    class_counts = torch.zeros(len(classes))
    for _, label in dataset:
        class_counts += label  # label is one-hot encoded

    # Calculate weights (inverse of frequency)
    eps = 1e-5
    weights = 1.0 / (class_counts + eps)

    # Normalize weights to sum to number of classes
    weights = weights * (len(classes) / weights.sum())
    weights = weights.to(device)

    print(f"Class weights range: min={weights.min().item():.4f}, max={weights.max().item():.4f}")
    print(f"Class weights sum: {weights.sum().item():.4f}")

    def weighted_loss(outputs, targets):
        # outputs: [batch_size, num_classes]
        # targets: [batch_size, num_classes] (one-hot)

        # Get probabilities
        probs = torch.nn.functional.softmax(outputs, dim=1)
        log_probs = torch.nn.functional.log_softmax(outputs, dim=1)

        # Basic negative log likelihood per sample
        nll_loss = -(targets * log_probs)  # [batch_size, num_classes]

        # Apply class weights
        weighted_loss = weights.unsqueeze(0) * nll_loss  # [batch_size, num_classes]
        base_loss = weighted_loss.sum(dim=1).mean()

        # Calculate confidence penalty
        # This penalizes the model for being too confident in its predictions
        max_probs, _ = probs.max(dim=1)
        confidence_loss = -torch.log(1.01 - max_probs).mean()  # Add 0.01 to avoid log(0)

        # Calculate entropy penalty
        # This encourages the model to be uncertain about non-target classes
        entropy = -(probs * log_probs).sum(dim=1).mean()
        entropy_bonus = -confidence_penalty * entropy  # Negative because we want to maximize entropy

        # Combine losses
        total_loss = base_loss + confidence_penalty * confidence_loss + entropy_bonus

        # Add minimal L2 regularization
        l2_reg = 0.001 * sum(p.pow(2.0).sum() for p in outputs)

        return total_loss + l2_reg

    return weighted_loss