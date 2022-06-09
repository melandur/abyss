import monai
from monai import transforms as tf


class Augmentation:
    """Define augmentation transformations for train and val set"""

    def __init__(self, config_manager):
        params = config_manager.params
        monai.utils.set_determinism(params['meta']['seed'])

        self.train_transforms = tf.Compose(
            [
                tf.RandGaussianNoise(),
                tf.RandGaussianSmooth(),
                tf.RandScaleIntensity(),
                tf.RandFlip(),
                tf.RandAdjustContrast(),
                tf.RandRotate(),
                tf.RandScaleCrop(),
            ]
        )

        self.val_transforms = tf.Compose(
            [
                tf.RandGaussianNoise(),
                tf.RandGaussianSmooth(),
                tf.RandScaleIntensity(),
                tf.RandFlip(),
                tf.RandAdjustContrast(),
                tf.RandRotate(),
                tf.RandScaleCrop(),
            ]
        )
