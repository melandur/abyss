from typing import ClassVar

import numpy as np
from monai import transforms as tf


class Augmentation:
    """Composes transformation based on the config file order"""

    def __init__(self, config_manager: ClassVar):
        self.params = config_manager.get_params()

    def compose_transforms(self) -> tf.Transform:
        """Compose active transformation"""
        transforms = []
        for tr_name, tr_params in self.params['augmentation'].items():
            if tr_name == 'RandGaussianNoise':
                transforms.append(tf.RandGaussianNoise(**tr_params, dtype=np.float32))
            elif tr_name == 'RandGaussianSmooth':
                transforms.append(tf.RandGaussianSmooth(**tr_params))
            elif tr_name == 'RandScaleIntensity':
                transforms.append(tf.RandScaleIntensity(**tr_params, dtype=np.float32))
            elif tr_name == 'RandFlip':
                transforms.append(tf.RandFlip(**tr_params))
            elif tr_name == 'RandAdjustContrast':
                transforms.append(tf.RandAdjustContrast(**tr_params))
            elif tr_name == 'RandRotate':
                transforms.append(tf.RandRotate(**tr_params, dtype=np.float64))
            elif tr_name == 'RandScaleCrop':
                transforms.append(tf.RandScaleCrop(**tr_params))
            elif tr_name == 'RandHistogramShift':
                transforms.append(tf.RandHistogramShift(**tr_params))
            elif tr_name == 'RandSpatialCrop':
                transforms.append(tf.RandSpatialCrop(**tr_params))
            elif tr_name == 'RandBiasField':
                transforms.append(tf.RandBiasField(**tr_params, dtype=np.float32))
            elif tr_name == 'RandRotate90':
                transforms.append(tf.RandRotate90(**tr_params))
            else:
                raise NotImplementedError(f'{tr_name} is missing, add transformation -> training -> augmentation.py')
        return tf.Compose(transforms)

    def __repr__(self):
        return f'{self.__class__.__name__}, order of used transforms -> {list(self.params["augmentation"].keys())}'


if __name__ == '__main__':
    from abyss.config import ConfigManager

    cm = ConfigManager()
    aug = Augmentation(cm)
    composed_transformation = aug.compose_transforms()
    print(composed_transformation)
    print(aug)
