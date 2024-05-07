from config import ConfigFile

from training.trainer import get_trainer
from training.model import Model

config_file = ConfigFile()
config = config_file.get_config()

model = Model(config)
trainer = get_trainer(config)

if config['mode']['train']:
    trainer.fit(model)

if config['mode']['test']:
    trainer.test(model)
