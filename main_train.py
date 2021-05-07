import sys
from loguru import logger as log

from main_conf import ConfigManager
from src.dataset.dataset_init_path_scan import DataSetInitPathScan
from src.pre_processing.pre_processing import PreProcessing
from src.training.training import Training

if __name__ == '__main__':
    params = ConfigManager(load_conf_file_path=None).params
    log.remove()  # fresh start
    log.add(sys.stderr, level=params['logger']['level'])

    if params['pipeline_steps']['dataset']:
        log.info('Started with dataset preparation:')

    if params['pipeline_steps']['pre_processing']:
        log.info('Started with pre-processing:')
        ds_init_path_scan = DataSetInitPathScan(params)
        pre_processing = PreProcessing(params, ds_init_path_scan.data_path_store)

    if params['pipeline_steps']['training']:
        log.info('Started with training:')
        Training(params)

    if params['pipeline_steps']['post_processing']:
        log.info('Started with post-processing:')


    # preprocesse and and slplit data to separeate folders.

    # create a monai data set

    # define transformation

    # train this shit

    # prepare data
    # start train
