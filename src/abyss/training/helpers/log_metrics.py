import torch
import torchmetrics


def log_accuracy(self, output: torch.Tensor, label: torch.Tensor, stage: str = '') -> None:
    """Log accuracy for certain stage"""
    x = torchmetrics.functional.classification.accuracy(output.to(torch.float32), label.to(torch.int8))
    self.log(f'{stage}_acc', x, prog_bar=True, on_epoch=True)


def log_dice(self, output: torch.Tensor, label: torch.Tensor, stage: str = '') -> None:
    """Log dice for certain stage"""
    x = torchmetrics.functional.classification.dice(output.to(torch.float32), label.to(torch.int8))
    self.log(f'{stage}_dice', x, prog_bar=True, on_epoch=True)
