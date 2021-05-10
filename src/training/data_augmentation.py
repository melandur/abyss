import numpy as np
from monai import transforms as tf


class ConvertToMultiChannelBasedOnBratsClassesd(tf.MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(np.logical_or(np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d


class DataAugmentation:

    def __init__(self, params):
        self.params = params

        self.train_transform = tf.Compose(
            [
                # load 4 Nifti images and stack them together
                tf.LoadImaged(keys=['image', 'label']),
                tf.AsChannelFirstd(keys=['image', 'label']),
                tf.AddChanneld(keys='image'),
                ConvertToMultiChannelBasedOnBratsClassesd(keys='label'),
                tf.Spacingd(keys=['image', 'label'],
                            pixdim=(1.0, 1.0, 1.0),
                            mode=('bilinear', 'nearest')),
                tf.Orientationd(keys=['image', 'label'], axcodes='RAS'),
                tf.RandSpatialCropd(keys=['image', 'label'], roi_size=[128, 128, 64], random_size=False),
                tf.RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
                tf.NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
                tf.RandScaleIntensityd(keys='image', factors=0.1, prob=0.5),
                tf.RandShiftIntensityd(keys='image', offsets=0.1, prob=0.5),
                tf.ToTensord(keys=['image', 'label']),
            ]
        )

        self.val_transform = tf.Compose(
            [
                tf.LoadImaged(keys=["image", "label"]),
                tf.AsChannelFirstd(keys=['image', 'label']),
                tf.AddChanneld(keys='image'),
                ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                tf.Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                tf.Orientationd(keys=["image", "label"], axcodes="RAS"),
                tf.CenterSpatialCropd(keys=["image", "label"], roi_size=[128, 128, 64]),
                tf.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                tf.ToTensord(keys=["image", "label"]),
            ]
        )

        self.test_transform = tf.Compose(
            [
                tf.LoadImaged(keys=['image', 'label']),
                tf.AsChannelFirstd(keys='image'),
                ConvertToMultiChannelBasedOnBratsClassesd(keys='label'),
                tf.Spacingd(keys=['image', 'label'],
                            pixdim=(1.0, 1.0, 1.0),
                            mode=('bilinear', 'nearest')),
                tf.Orientationd(keys=['image', 'label'], axcodes='RAS'),
                tf.CenterSpatialCropd(keys=['image', 'label'], roi_size=[128, 128, 64]),
                tf.NormalizeIntensityd(keys='image', nonzero=True, channel_wise=True),
                tf.ToTensord(keys=['image', 'label'])
            ]
        )


if __name__ == '__main__':
    data_dicts = {'image': r'C:\Users\melandur\Desktop\mo\my_test\imagesTr\BraTS19_2013_3_1_flair.nii.gz',
                  'label': r'C:\Users\melandur\Desktop\mo\my_test\labelsTr\BraTS19_2013_3_1_seg.nii.gz'}

    da = DataAugmentation()
    x = da.train_transform(data_dicts)
    print('image shape:', np.shape(x['image']))
    print('label shape:', np.shape(x['label']))
