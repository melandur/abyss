import os
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

    ckpt_path = None
    if config['training']['reload_checkpoint']:
        if os.path.exists(config['project']['results_path']):
            ckpt_path = os.path.join(config['project']['results_path'], 'last.ckpt')
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f'No last found in -> {config["project"]["results_path"]}')

    trainer.fit(model, ckpt_path=ckpt_path)

if config['mode']['test']:
    trainer.test(model)
