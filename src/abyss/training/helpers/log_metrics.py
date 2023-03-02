import torch
from monai.metrics.meandice import DiceMetric


def log(self, output: torch.Tensor, label: torch.Tensor, stage: str = '') -> None:
    """Log metrics"""
    if 'dice' in self.params['training']['log_metrics']:
        output_cpu = output.cpu().detach().numpy()
        label_cpu = label.cpu().detach().numpy()
        print(output_cpu.shape, label_cpu.shape)
        dice = DiceMetric()
        x = dice(output, label)
        self.log(f'{stage}_dice', x, prog_bar=True, on_step=False, on_epoch=True)

    if 'accuracy' in self.params['training']['log_metrics']:
        # x = torchmetrics.functional.classification.accuracy(output, label, task='multiclass', num_classes=3)
        self.log(f'{stage}_acc', x, prog_bar=True, on_step=False, on_epoch=True)
