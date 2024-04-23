from abyss.config import ConfigManager
from abyss.training.trainer import Trainer


class Training(ConfigManager):
    """Runs according to the config file"""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._shared_state.update(kwargs)
        self.trainer = Trainer()

    def __call__(self) -> None:
        if any(self.params['pipeline_steps']['training'].values()):

            if self.params['pipeline_steps']['training']['fit']:
                self.trainer.setup('fit')

            if self.params['pipeline_steps']['training']['test']:
                self.trainer.setup('test')
