# import numpy as np
# # from monai import transforms as tf
#
#
# class ConvertToMultiChannelBasedOnBratsClassesd(tf.MapTransform):
#     """
#     Convert labels to multi channels based on brats classes:
#     label 1 is the peritumoral edema
#     label 2 is the GD-enhancing tumor
#     label 3 is the necrotic and non-enhancing tumor core
#     The possible classes are TC (Tumor core), WT (Whole tumor)
#     and ET (Enhancing tumor).
#
#     """
#
#     def __call__(self, data):
#         tmp_dict = dict(data)
#         for key in self.keys:
#             result = []
#             # merge label 2 and label 3 to construct TC
#             result.append(np.logical_or(tmp_dict[key] == 2, tmp_dict[key] == 3))
#             # merge labels 1, 2 and 3 to construct WT
#             result.append(np.logical_or(np.logical_or(tmp_dict[key] == 2, tmp_dict[key] == 3), tmp_dict[key] == 1))
#             # label 2 is ET
#             result.append(tmp_dict[key] == 2)
#             tmp_dict[key] = np.stack(result, axis=0).astype(np.float32)
#         return tmp_dict


class DataAugmentation:
    """Apply data augmentation"""

    def __init__(self):
        # self.params = params

        self.train_transform = tf.Compose(
            [
                # load 4 Nifti images and stack them together
                tf.LoadImaged(keys=['data', 'label']),
                tf.AsChannelFirstd(keys=['data', 'label']),
                tf.AddChanneld(keys='data'),
                ConvertToMultiChannelBasedOnBratsClassesd(keys='label'),
                tf.Spacingd(keys=['data', 'label'], pixdim=(1.0, 1.0, 1.0), mode=('bilinear', 'nearest')),
                tf.Orientationd(keys=['data', 'label'], axcodes='RAS'),
                tf.RandSpatialCropd(keys=['data', 'label'], roi_size=[128, 128, 64], random_size=False),
                tf.RandFlipd(keys=['data', 'label'], prob=0.5, spatial_axis=0),
                tf.NormalizeIntensityd(keys='data', nonzero=True, channel_wise=True),
                tf.RandScaleIntensityd(keys='data', factors=0.1, prob=0.5),
                tf.RandShiftIntensityd(keys='data', offsets=0.1, prob=0.5),
                tf.ToTensord(keys=['data', 'label']),
            ]
        )

        self.val_transform = tf.Compose(
            [
                tf.LoadImaged(keys=["image", "label"]),
                tf.AsChannelFirstd(keys=['data', 'label']),
                tf.AddChanneld(keys='data'),
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
                tf.LoadImaged(keys=['data', 'label']),
                tf.AsChannelFirstd(keys='data'),
                ConvertToMultiChannelBasedOnBratsClassesd(keys='label'),
                tf.Spacingd(keys=['data', 'label'], pixdim=(1.0, 1.0, 1.0), mode=('bilinear', 'nearest')),
                tf.Orientationd(keys=['data', 'label'], axcodes='RAS'),
                tf.CenterSpatialCropd(keys=['data', 'label'], roi_size=[128, 128, 64]),
                tf.NormalizeIntensityd(keys='data', nonzero=True, channel_wise=True),
                tf.ToTensord(keys=['data', 'label']),
            ]
        )

        # self.debug_transform = tf.Compose(
        #     [
        #         ConcatenateImages(keys=['data'])
        #     ]
        # )
        #


if __name__ == '__main__':
    PATH = 'C:\\Users\\melandur\\Downloads\\mytest\\BratsExp1\\test1\\structured_dataset'
    data_dicts = {
        "image": {
            "ACRINDSCMRBrain0571": {
                "t1": f"{PATH}\\image\\ACRINDSCMRBrain0571_t1.nii.gz",
                "t1ce": f"{PATH}\\image\\ACRINDSCMRBrain0571_t1ce.nii.gz",
                "flair": f"{PATH}\\image\\ACRINDSCMRBrain0571_flair.nii.gz",
                "t2": f"{PATH}\\image\\ACRINDSCMRBrain0571_t2.nii.gz",
            },
            "ACRINDSCMRBrain0621": {
                "t1": f"{PATH}\\image\\ACRINDSCMRBrain0621_t1.nii.gz",
                "t1ce": f"{PATH}\\image\\ACRINDSCMRBrain0621_t1ce.nii.gz",
                "flair": f"{PATH}\\image\\ACRINDSCMRBrain0621_flair.nii.gz",
                "t2": f"{PATH}\\image\\ACRINDSCMRBrain0621_t2.nii.gz",
            },
            "ACRINDSCMRBrain0641": {
                "t1": f"{PATH}\\image\\ACRINDSCMRBrain0641_t1.nii.gz",
                "t1ce": f"{PATH}\\image\\ACRINDSCMRBrain0641_t1ce.nii.gz",
                "flair": f"{PATH}\\image\\ACRINDSCMRBrain0641_flair.nii.gz",
                "t2": f"{PATH}\\image\\ACRINDSCMRBrain0641_t2.nii.gz",
            },
            "ACRINDSCMRBrain0671": {
                "t1": f"{PATH}\\image\\ACRINDSCMRBrain0671_t1.nii.gz",
                "t1ce": f"{PATH}\\image\\ACRINDSCMRBrain0671_t1ce.nii.gz",
                "flair": f"{PATH}\\image\\ACRINDSCMRBrain0671_flair.nii.gz",
                "t2": f"{PATH}\\image\\ACRINDSCMRBrain0671_t2.nii.gz",
            },
            "ACRINDSCMRBrain0741": {
                "t1": f"{PATH}\\image\\ACRINDSCMRBrain0741_t1.nii.gz",
                "t1ce": f"{PATH}\\image\\ACRINDSCMRBrain0741_t1ce.nii.gz",
                "flair": f"{PATH}\\image\\ACRINDSCMRBrain0741_flair.nii.gz",
                "t2": f"{PATH}\\image\\ACRINDSCMRBrain0741_t2.nii.gz",
            },
        },
        "label": {
            "ACRINDSCMRBrain0571": f"{PATH}\\label\\ACRINDSCMRBrain0571_seg.nii.gz",
            "ACRINDSCMRBrain0621": f"{PATH}\\label\\ACRINDSCMRBrain0621_seg.nii.gz",
            "ACRINDSCMRBrain0641": f"{PATH}\\label\\ACRINDSCMRBrain0641_seg.nii.gz",
            "ACRINDSCMRBrain0671": f"{PATH}\\label\\ACRINDSCMRBrain0671_seg.nii.gz",
            "ACRINDSCMRBrain0741": f"{PATH}\\label\\ACRINDSCMRBrain0741_seg.nii.gz",
        },
    }
    from abyss.config import ConfigManager

    # from abyss.pre_processing.pre_processing_helpers import ConcatenateImages

    params = ConfigManager().params
    da = DataAugmentation()

    x = da.debug_transform(data_dicts)
    print(x['data'].keys())
    # for case in x['data'].keys():
    #     print(np.shape(x['data'][case]))

    # print(x)
    # print(x)
    #
    # for case in data_dicts['data']:
    #     print(case)
    #     print()
    # x = tf.ConcatItemsD()
    #
    #     print(x)

    # print(x)
    # print('image shape:', np.shape(x['data']))
    # print('label shape:', np.shape(x['label']))
