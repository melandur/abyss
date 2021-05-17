import sys
from loguru import logger as log

from main_conf import ConfigManager
from src.dataset.dataset_init_path_scan import DataSetInitPathScan
from src.pre_processing.pre_processing import PreProcessing
from src.training.train_dataset_path_scan import TrainDataSetPathScan
from src.training.training import Training


if __name__ == '__main__':

    cm = ConfigManager()
    cm.load_config_file(file_path=None)

    log.remove()  # fresh start
    log.add(sys.stderr, level=cm.params['logger']['level'])

    if cm.params['pipeline_steps']['dataset']:
        log.info('Started with dataset preparation:')
        DataSetInitPathScan(cm)
        # StructureDataSet(params)

    if cm.params['pipeline_steps']['pre_processing']:
        log.info('Started with pre-processing:')
        PreProcessing(cm)

    if cm.params['pipeline_steps']['training']:
        log.info('Started with training:')
        TrainDataSetPathScan(cm)
        Training(cm)

    if cm.params['pipeline_steps']['post_processing']:
        log.info('Started with post-processing:')


    # preprocesse and and slplit data to separeate folders.

    # create a monai data set

    # define transformation

    # train this shit

    # prepare data
    # start train
