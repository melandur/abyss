import resource

from config import ConfigFile
from training.model import Model
from training.trainer import get_trainer

# Increase the number of file descriptors to the maximum allowed
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


config_file = ConfigFile()
config = config_file.get_config()
model = Model(config)
trainer = get_trainer(config)

if config['mode']['train']:
    trainer.fit(model)

if config['mode']['test']:
    trainer.test(model)
