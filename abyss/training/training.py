from abyss.training.model import Model
from abyss.training.trainer import Trainer


class Training:
    """That's were the gpu is getting sweaty"""

    def __init__(self, config_manager):
        self.config_manager = config_manager

    def __call__(self):
        model = Model(self.config_manager)
        trainer = Trainer(self.config_manager)()
        trainer.fit(model)
