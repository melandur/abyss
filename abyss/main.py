import time
from loguru import logger

from abyss.config import ConfigManager
from abyss.dataset import DataCleaner
from abyss.pre_processing import PreProcessing
from abyss.training import CreateTrainset, Training

if __name__ == '__main__':
    start = time.time()
    cm = ConfigManager(load_config_file_path='/home/melandur/Downloads/Abyss_test/experiment_1/config_data/config.json')

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

    logger.info(f'Execution time: {time.time() - start}')
