import math


class LearningRateScheduler:
    """Learning rate scheduler"""

    def __init__(self, optimizer, lr_start: float, lr_end: float, warmup_epochs: int, total_epochs: int) -> None:
        self.optimizer = optimizer
        self.lr_start = lr_start
        self.lr_end = lr_end
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs

    def step(self, epoch: int) -> float:
        """Update optimizer learning rate for each epoch"""
        if epoch < self.warmup_epochs:  # cosine annealing warmup
            lr = (
                self.lr_start + (self.lr_end - self.lr_start) * (1 - math.cos(epoch / self.warmup_epochs * math.pi)) / 2
            )
        else:  # poly decay
            epoch = epoch - self.warmup_epochs
            lr_current = self.optimizer.param_groups[0]['lr']
            lr = lr_current * (1 - epoch / (self.total_epochs - self.warmup_epochs)) ** 0.9

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

    def state_dict(self):
        """Return state dict"""
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Load state dict"""
        self.__dict__.update(state_dict)
