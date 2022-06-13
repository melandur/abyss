from datetime import datetime

from loguru import logger

from abyss import ConfigManager, DataReader, CreateHDF5, PreProcessing, Training


if __name__ == '__main__':
    start = datetime.now()
    cm = ConfigManager(load_config_file_path=None)

    if cm.params['pipeline_steps']['data_reader']:
        DataReader(cm)()

    if cm.params['pipeline_steps']['pre_processing']:
        PreProcessing(cm)()

    if cm.params['pipeline_steps']['create_trainset']:
        CreateHDF5(cm)()

    if cm.params['pipeline_steps']['training']:
        Training(cm)()

    logger.info(f'Execution time -> {datetime.now() - start}')
