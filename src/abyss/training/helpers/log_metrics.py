import torch
import torchmetrics


def log_accuracy(self, output: torch.Tensor, label: torch.Tensor, stage: str = '') -> None:
    """Log accuracy for certain stage"""
    x = torchmetrics.functional.classification.accuracy(output.to(torch.float32), label.to(torch.uint8))
    self.log(f'{stage}_acc', x, prog_bar=True, on_step=False, on_epoch=True)


def log_dice(self, output: torch.Tensor, label: torch.Tensor, stage: str = '') -> None:
    """Log dice for certain stage"""
    output = torch.sigmoid(output) > 0.5  # threshold
    x = torchmetrics.functional.classification.dice(output, label)
    self.log(f'{stage}_dice', x, prog_bar=True, on_step=False, on_epoch=True)
