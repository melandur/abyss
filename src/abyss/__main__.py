import os
from datetime import datetime

from loguru import logger

from abyss import (
    ConfigManager,
    CreateHDF5,
    DataReader,
    ExtractWeights,
    Inference,
    PostProcessing,
    PreProcessing,
    Training,
)

os.environ['OMP_NUM_THREADS'] = '1'


def run_pipeline():
    """Start pipeline"""
    start = datetime.now()

    config_manager = ConfigManager(load_config_file_path=None)
    data_reader = DataReader()
    pre_processing = PreProcessing()
    create_hdf5 = CreateHDF5()
    training = Training()
    extract_weights = ExtractWeights()
    inference = Inference()
    post_processing = PostProcessing()

    config_manager()
    if config_manager.params['pipeline_steps']['data_reader']:
        data_reader()
    if config_manager.params['pipeline_steps']['pre_processing']:
        pre_processing()
    if config_manager.params['pipeline_steps']['create_trainset']:
        create_hdf5()
    if config_manager.params['pipeline_steps']['training']:
        training()

    if config_manager.params['pipeline_steps']['production']['extract_weights']:
        extract_weights()
    if config_manager.params['pipeline_steps']['production']['inference']:
        inference()
    if config_manager.params['pipeline_steps']['production']['post_processing']:
        post_processing()

    logger.info(f'Execution time -> {datetime.now() - start}')


if __name__ == '__main__':
    run_pipeline()
