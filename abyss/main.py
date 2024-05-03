from config import ConfigFile

from abyss.training.train import train, validation

config_file = ConfigFile()
config = config_file.get_config()

if config['mode']['train']:
    train(config)

if config['mode']['test']:
    validation(config)
