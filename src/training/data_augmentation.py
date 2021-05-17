import torch
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

    def __init__(self):
        # self.params = params

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

        self.debug_transform = tf.Compose(
            [
                ConcatenateImages(keys=['image'])
            ]
        )





if __name__ == '__main__':
    data_dicts = {
        "image": {
            "ACRINDSCMRBrain0571": {
                "t1": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\image\\ACRINDSCMRBrain0571_t1.nii.gz",
                "t1ce": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\image\\ACRINDSCMRBrain0571_t1ce.nii.gz",
                "flair": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\image\\ACRINDSCMRBrain0571_flair.nii.gz",
                "t2": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\image\\ACRINDSCMRBrain0571_t2.nii.gz"
            },
            "ACRINDSCMRBrain0621": {
                "t1": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\image\\ACRINDSCMRBrain0621_t1.nii.gz",
                "t1ce": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\image\\ACRINDSCMRBrain0621_t1ce.nii.gz",
                "flair": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\image\\ACRINDSCMRBrain0621_flair.nii.gz",
                "t2": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\image\\ACRINDSCMRBrain0621_t2.nii.gz"
            },
            "ACRINDSCMRBrain0641": {
                "t1": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\image\\ACRINDSCMRBrain0641_t1.nii.gz",
                "t1ce": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\image\\ACRINDSCMRBrain0641_t1ce.nii.gz",
                "flair": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\image\\ACRINDSCMRBrain0641_flair.nii.gz",
                "t2": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\image\\ACRINDSCMRBrain0641_t2.nii.gz"
            },
            "ACRINDSCMRBrain0671": {
                "t1": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\image\\ACRINDSCMRBrain0671_t1.nii.gz",
                "t1ce": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\image\\ACRINDSCMRBrain0671_t1ce.nii.gz",
                "flair": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\image\\ACRINDSCMRBrain0671_flair.nii.gz",
                "t2": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\image\\ACRINDSCMRBrain0671_t2.nii.gz"
            },
            "ACRINDSCMRBrain0741": {
                "t1": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\image\\ACRINDSCMRBrain0741_t1.nii.gz",
                "t1ce": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\image\\ACRINDSCMRBrain0741_t1ce.nii.gz",
                "flair": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\image\\ACRINDSCMRBrain0741_flair.nii.gz",
                "t2": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\image\\ACRINDSCMRBrain0741_t2.nii.gz"
            }
        },
        "label": {
                "ACRINDSCMRBrain0571": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\label\\ACRINDSCMRBrain0571_seg.nii.gz",
                "ACRINDSCMRBrain0621": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\label\\ACRINDSCMRBrain0621_seg.nii.gz",
                "ACRINDSCMRBrain0641": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\label\\ACRINDSCMRBrain0641_seg.nii.gz",
                "ACRINDSCMRBrain0671": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\label\\ACRINDSCMRBrain0671_seg.nii.gz",
                "ACRINDSCMRBrain0741": "C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset\\label\\ACRINDSCMRBrain0741_seg.nii.gz"
            }
    }
    from main_conf import ConfigManager
    from src.pre_processing.pre_processing_helpers import ConcatenateImages

    params = ConfigManager().params
    da = DataAugmentation()

    x = da.debug_transform(data_dicts)
    print(x['image'].keys())
    # for case in x['image'].keys():
    #     print(np.shape(x['image'][case]))




    # print(x)
    # print(x)
    #
    # for case in data_dicts['image']:
    #     print(case)
    #     print()
        # x = tf.ConcatItemsD()
    #
    #     print(x)

    # print(x)
    # print('image shape:', np.shape(x['image']))
    # print('label shape:', np.shape(x['label']))
