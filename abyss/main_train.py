from loguru import logger

from abyss.config import ConfigManager
from abyss.dataset.dataset_analyzer import DatasetAnalyzer
from abyss.pre_processing.pre_processing import PreProcessing
from abyss.training.create_train_datasets import CreateTrainDatasets
from abyss.training.training import Training

if __name__ == '__main__':

    cm = ConfigManager(load_config_file_path=None)

    if cm.params['pipeline_steps']['read_dataset']:
        DatasetAnalyzer(cm)

    if cm.params['pipeline_steps']['pre_processing']:
        PreProcessing(cm)

    if cm.params['pipeline_steps']['create_datasets']:
        CreateTrainDatasets(cm)

    if cm.params['pipeline_steps']['training']:
        Training(cm)

    if cm.params['pipeline_steps']['post_processing']:
        logger.info('Started with post-processing:')
