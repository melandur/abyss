from loguru import logger as log

from main_conf import ConfigManager
from src.dataset.dataset_init_path_scan import DataSetInitPathScan
from src.pre_processing.pre_processing import PreProcessing
from src.training.create_datasets import CreateDatasets
from src.training.training import Training


if __name__ == '__main__':

    cm = ConfigManager(load_config_file_path=None)

    if cm.params['pipeline_steps']['read_dataset']:
        log.info('Started with dataset preparation:')
        DataSetInitPathScan(cm)

    if cm.params['pipeline_steps']['pre_processing']:
        log.info('Started with pre-processing:')
        PreProcessing(cm)

    if cm.params['pipeline_steps']['create_datasets']:
        log.info('Started with train data set creation:')
        CreateDatasets(cm)

    if cm.params['pipeline_steps']['training']:
        log.info('Started with training:')
        Training(cm)

    if cm.params['pipeline_steps']['post_processing']:
        log.info('Started with post-processing:')


    # preprocesse and and slplit data to separeate folders.

    # create a monai data set

    # define transformation

    # train this shit

    # prepare data
    # start train
