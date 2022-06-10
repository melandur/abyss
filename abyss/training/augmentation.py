from typing import ClassVar

import monai
import numpy as np
from monai import transforms as tf


class Augmentation:
    """Composes transformation based on the config file order"""

    def __init__(self, config_manager: ClassVar):
        self.params = config_manager.params
        monai.utils.set_determinism(self.params['meta']['seed'])

    def compose_transforms(self) -> tf.Transform:
        transforms = []
        for trans_name, trans_params in self.params['augmentation'].items():
            if trans_name == 'RandGaussianNoise':
                transforms.append(tf.RandGaussianNoise(**trans_params, dtype=np.float32))
            elif trans_name == 'RandGaussianSmooth':
                transforms.append(tf.RandGaussianSmooth(**trans_params))
            elif trans_name == 'RandScaleIntensity':
                transforms.append(tf.RandScaleIntensity(**trans_params, dtype=np.float32))
            elif trans_name == 'RandFlip':
                transforms.append(tf.RandFlip(**trans_params))
            elif trans_name == 'RandAdjustContrast':
                transforms.append(tf.RandAdjustContrast(**trans_params))
            elif trans_name == 'RandRotate':
                transforms.append(tf.RandRotate(**trans_params, dtype=np.float64))
            elif trans_name == 'RandScaleCrop':
                transforms.append(tf.RandScaleCrop(**trans_params))
            else:
                raise NotImplementedError(f'{trans_name} is missing, add transformation -> training -> augmentation.py')
        return tf.Compose(transforms)


if __name__ == '__main__':
    from abyss.config import ConfigManager

    cm = ConfigManager()
    aug = Augmentation(cm)
    composed_transformation = aug.compose_transforms()
    print(composed_transformation)
