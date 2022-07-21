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


class Pipeline:
    """Runs according to the config file"""

    def __init__(self):
        config_manager = ConfigManager(load_config_file_path=None)
        self.data_reader = DataReader()
        self.pre_processing = PreProcessing()
        self.create_hdf5 = CreateHDF5()
        self.training = Training()
        self.extract_weights = ExtractWeights()
        self.inference = Inference()
        self.post_processing = PostProcessing()
        config_manager()

    def __call__(self):
        start = datetime.now()
        self.data_reader()
        self.pre_processing()
        self.create_hdf5()
        self.training()

        self.extract_weights()
        self.inference()
        self.post_processing()
        logger.info(f'Execution time -> {datetime.now() - start}')


if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline()
