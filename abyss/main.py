from loguru import logger

from abyss.config import ConfigManager
from abyss.dataset import DataCleaner
from abyss.pre_processing import PreProcessing
from abyss.training import CreateTrainset, Training

if __name__ == '__main__':
    import time

    tic = time.time()
    cm = ConfigManager(load_config_file_path=None)

    if cm.params['pipeline_steps']['clean_dataset']:
        DataCleaner(cm)()

    if cm.params['pipeline_steps']['pre_processing']:
        PreProcessing(cm)

    if cm.params['pipeline_steps']['create_trainset']:
        CreateTrainset(cm)

    if cm.params['pipeline_steps']['training']:
        Training(cm)

    if cm.params['pipeline_steps']['post_processing']:
        logger.info('Started with post-processing:')
    print(time.time() - tic)
