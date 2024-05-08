from config import ConfigFile
from training.model import Model
from training.trainer import get_trainer

config_file = ConfigFile()
config = config_file.get_config()

model = Model(config)
trainer = get_trainer(config)

if config['mode']['train']:
    trainer.fit(model)

if config['mode']['test']:
    trainer.test(model)
