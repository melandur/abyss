from torch.optim.lr_scheduler import _LRScheduler


class WarmupPolyLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        base_lr: float,
        total_steps: int,
        warmup_steps: int,
        exponent: float = 0.9,
        last_epoch: int = -1,
    ) -> None:
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.exponent = exponent
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step <= self.warmup_steps:  # Linear warmup
            scale = step / max(1, self.warmup_steps)
            return [self.base_lr * scale for _ in self.optimizer.param_groups]

        # Polynomial decay
        current_step = step - self.warmup_steps
        progress = current_step / max(1, self.total_steps)
        decay = (1 - progress) ** self.exponent
        return [self.base_lr * decay for _ in self.optimizer.param_groups]

