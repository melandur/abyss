import numpy as np
import torch
from monai.data import decollate_batch
from monai.metrics import DiceMetric

from abyss.config import ConfigManager


class Metrics(ConfigManager):

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._shared_state.update(kwargs)
        self.labels = self.params['training']['output_channels']
        self._state = {}
        self._used_metrics = {}

    def calculate(self, output: torch.Tensor, label: torch.Tensor, tag: str) -> None:
        """Calculate metrics for certain stage"""
        output = torch.sigmoid(output)
        output = (output > 0.5).int()

        for metric in self.params['trainer']['metrics']:
            if hasattr(self, metric):
                metric_method = getattr(self, metric)
                metric_method(output, label, tag)

    def aggregate(self, tag: str) -> None:
        """Aggregate metrics for certain stage"""
        for key, metric in self._used_metrics.items():
            results = np.round(metric[tag].aggregate().cpu().numpy(), 3)
            self._state[key] = {tag: {}}
            for idx, label in enumerate(self.labels):
                self._state[key][tag].update({label: results[idx]})
            metric[tag].reset()

    def dice_score(self, outputs: torch.Tensor, labels: torch.Tensor, tag: str) -> None:
        """Log dice for certain stage"""
        self._used_metrics['dice'] = {
            tag: DiceMetric(include_background=True, reduction='mean_batch', num_classes=len(self.labels))
        }
        outputs_list = decollate_batch(outputs)
        labels_list = decollate_batch(labels)
        self._used_metrics['dice'][tag](outputs_list, labels_list)

    def get(self, metric: str, tag: str) -> float:
        """Get metric value"""
        if metric not in self._state:
            raise ValueError(f"Metric {metric} not found in state")
        if tag not in self._state[metric]:
            raise ValueError(f"Tag {tag} not found in metric {metric}")
        return self._state[metric][tag]
