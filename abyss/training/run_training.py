from abyss.training.trainer import Trainer
from abyss.training.model import Model

model = Model()
# data_module = DataModule()
trainer = Trainer()
trainer.fit(model, data_module)