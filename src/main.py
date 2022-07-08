import os
from datetime import datetime

from loguru import logger

from abyss import ConfigManager, CreateHDF5, DataReader, PreProcessing, Training

os.environ['OMP_NUM_THREADS'] = '1'

if __name__ == '__main__':
    start = datetime.now()

    cm = ConfigManager(load_config_file_path=None)
    data_reader = DataReader()
    pre_processing = PreProcessing()
    create_hdf5 = CreateHDF5()
    training = Training()

    cm()
    if cm.params['pipeline_steps']['data_reader']:
        data_reader()
    if cm.params['pipeline_steps']['pre_processing']:
        pre_processing()
    if cm.params['pipeline_steps']['create_trainset']:
        create_hdf5()
    if cm.params['pipeline_steps']['training']:
        training()

    logger.info(f'Execution time -> {datetime.now() - start}')
