import torch
import torchmetrics


def log(self, output: torch.Tensor, label: torch.Tensor, stage: str = '') -> None:
    """Log metrics"""
    if 'dice' in self.params['training']['log_metrics']:
        x = torchmetrics.functional.classification.dice(output.to(torch.float32), label.to(torch.uint8))
        self.log(f'{stage}_dice', x, prog_bar=True, on_step=False, on_epoch=True)

    if 'accuracy' in self.params['training']['log_metrics']:
        x = torchmetrics.functional.classification.accuracy(output, label, task='multiclass', num_classes=3)
        self.log(f'{stage}_acc', x, prog_bar=True, on_step=False, on_epoch=True)
