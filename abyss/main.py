import os
import time

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

os.environ['OMP_NUM_THREADS'] = '1'  # TODO: check if this is necessary
start_time = time.time()


class Pipeline:
    """Runs according to the config file"""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.data_reader = DataReader()
        self.pre_processing = PreProcessing()
        self.create_hdf5 = CreateHDF5()
        self.training = Training()
        self.extract_weights = ExtractWeights()
        self.inference = Inference()
        self.post_processing = PostProcessing()

    def __call__(self):
        self.config_manager()
        self.data_reader()
        self.pre_processing()
        self.create_hdf5()
        self.training()

        # self.extract_weights()
        # self.inference()
        # self.post_processing()

        self.__execution_time()

    @staticmethod
    def __execution_time():
        """Prints the execution time of the pipeline"""
        end_time = time.time()
        execution_time = end_time - start_time
        hours = int(execution_time // 3600)
        minutes = int((execution_time % 3600) // 60)
        seconds = int(execution_time % 60)
        logger.info(f'Execution time -> {hours:02d}:{minutes:02d}:{seconds:02d}')


if __name__ == '__main__':
    pipeline = Pipeline()
    pipeline()
