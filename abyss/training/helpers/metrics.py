import torch
import torchmetrics


def metric_accuracy(output: torch.Tensor, label: torch.Tensor):
    """Log accuracy for certain stage"""
    return torchmetrics.functional.classification.accuracy(output.to(torch.float32), label.to(torch.int8))


def metric_dice(output: torch.Tensor, label: torch.Tensor):
    """Log dice for certain stage"""
    return torchmetrics.functional.classification.dice(output.to(torch.float32), label.to(torch.int8))
