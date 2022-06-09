from abyss.training.trainer import Trainer


class Training:
    """That's were the gpu is getting sweaty"""

    def __init__(self, config_manager):
        params = config_manager.params

    def __call__(self):
        model = Model()
        data_module = DataModule()
        trainer = Trainer()
        trainer.fit(model, data_module)
