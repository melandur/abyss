from loguru import logger as log

from src.config.config_manager import ConfigManager
from src.dataset.dataset_init_path_scan import DataSetInitPathScan
from src.pre_processing.pre_processing import PreProcessing
from src.training.create_datasets import CreateDatasets
from src.training.training import Training


if __name__ == '__main__':

    cm = ConfigManager(load_config_file_path=None)

    if cm.params['pipeline_steps']['read_dataset']:
        DataSetInitPathScan(cm)

    if cm.params['pipeline_steps']['pre_processing']:
        PreProcessing(cm)

    if cm.params['pipeline_steps']['create_datasets']:
        CreateDatasets(cm)

    if cm.params['pipeline_steps']['training']:
        Training(cm)

    if cm.params['pipeline_steps']['post_processing']:
        log.info('Started with post-processing:')