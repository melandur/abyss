import torch
import torchmetrics


def metric_accuracy(output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Log accuracy for certain stage"""
    return torchmetrics.functional.classification.accuracy(output, label)


def metric_dice(output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Log dice for certain stage"""
    return torchmetrics.functional.classification.dice(output.to(torch.int8), label.to(torch.int8))
