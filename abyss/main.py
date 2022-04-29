from loguru import logger

from abyss.config import ConfigManager
from abyss.dataset import DataCleaner
from abyss.pre_processing.pre_processing import PreProcessing
from abyss.training.create_trainset import CreateTrainset
from abyss.training.training import Training

if __name__ == '__main__':

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
