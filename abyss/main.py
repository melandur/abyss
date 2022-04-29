from datetime import datetime

from loguru import logger

from abyss.config import ConfigManager
from abyss.dataset import DataSelection
from abyss.pre_processing import PreProcessing
from abyss.training import CreateTrainset, Training

if __name__ == '__main__':
    start = datetime.now()
    cm = ConfigManager(load_config_file_path=None)

    if cm.params['pipeline_steps']['data_selection']:
        DataSelection(cm)()

    if cm.params['pipeline_steps']['pre_processing']:
        PreProcessing(cm)

    if cm.params['pipeline_steps']['create_trainset']:
        CreateTrainset(cm)

    if cm.params['pipeline_steps']['training']:
        Training(cm)

    logger.info(f'Execution time -> {datetime.now() - start}')
