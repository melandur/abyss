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

        self.test = tf.Compose(
            [
                # tf.LoadImaged['']
                # tf.ConcatItemsd()
                # tf.ConcatItemsd(keys=['flair', 't1'], name='harry', allow_missing_keys=True)
                # tf.LoadImaged(keys=['image', 'label']),
            ]
        )
        #


if __name__ == '__main__':
    data_dicts = {
        'image': {
            'BraTS19_2013_18_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_18_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_18_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_18_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_18_1_t2.nii.gz'},
            'BraTS19_2013_21_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_21_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_21_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_21_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_21_1_t2.nii.gz'},
            'BraTS19_2013_22_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_22_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_22_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_22_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_22_1_t2.nii.gz'},
            'BraTS19_2013_2_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_2_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_2_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_2_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_2_1_t2.nii.gz'},
            'BraTS19_2013_5_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_5_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_5_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_5_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_5_1_t2.nii.gz'},
            'BraTS19_2013_7_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_7_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_7_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_7_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_2013_7_1_t2.nii.gz'},
            'BraTS19_CBICA_ABB_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ABB_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ABB_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ABB_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ABB_1_t2.nii.gz'},
            'BraTS19_CBICA_ABE_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ABE_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ABE_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ABE_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ABE_1_t2.nii.gz'},
            'BraTS19_CBICA_ALN_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ALN_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ALN_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ALN_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ALN_1_t2.nii.gz'},
            'BraTS19_CBICA_AME_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AME_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AME_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AME_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AME_1_t2.nii.gz'},
            'BraTS19_CBICA_AOS_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AOS_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AOS_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AOS_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AOS_1_t2.nii.gz'},
            'BraTS19_CBICA_AOZ_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AOZ_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AOZ_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AOZ_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AOZ_1_t2.nii.gz'},
            'BraTS19_CBICA_AQN_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AQN_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AQN_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AQN_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AQN_1_t2.nii.gz'},
            'BraTS19_CBICA_AQT_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AQT_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AQT_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AQT_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AQT_1_t2.nii.gz'},
            'BraTS19_CBICA_ARW_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ARW_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ARW_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ARW_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ARW_1_t2.nii.gz'},
            'BraTS19_CBICA_ARZ_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ARZ_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ARZ_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ARZ_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ARZ_1_t2.nii.gz'},
            'BraTS19_CBICA_ASA_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ASA_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ASA_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ASA_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ASA_1_t2.nii.gz'},
            'BraTS19_CBICA_ASK_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ASK_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ASK_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ASK_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ASK_1_t2.nii.gz'},
            'BraTS19_CBICA_ASO_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ASO_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ASO_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ASO_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ASO_1_t2.nii.gz'},
            'BraTS19_CBICA_ATN_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ATN_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ATN_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ATN_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_ATN_1_t2.nii.gz'},
            'BraTS19_CBICA_AUQ_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AUQ_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AUQ_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AUQ_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AUQ_1_t2.nii.gz'},
            'BraTS19_CBICA_AUW_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AUW_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AUW_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AUW_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AUW_1_t2.nii.gz'},
            'BraTS19_CBICA_AWH_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AWH_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AWH_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AWH_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AWH_1_t2.nii.gz'},
            'BraTS19_CBICA_AWX_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AWX_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AWX_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AWX_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AWX_1_t2.nii.gz'},
            'BraTS19_CBICA_AXN_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AXN_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AXN_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AXN_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AXN_1_t2.nii.gz'},
            'BraTS19_CBICA_AXO_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AXO_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AXO_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AXO_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AXO_1_t2.nii.gz'},
            'BraTS19_CBICA_AZH_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AZH_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AZH_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AZH_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_AZH_1_t2.nii.gz'},
            'BraTS19_CBICA_BCL_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_BCL_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_BCL_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_BCL_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_BCL_1_t2.nii.gz'},
            'BraTS19_CBICA_BGN_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_BGN_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_BGN_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_BGN_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_BGN_1_t2.nii.gz'},
            'BraTS19_CBICA_BGX_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_BGX_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_BGX_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_BGX_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_BGX_1_t2.nii.gz'},
            'BraTS19_CBICA_BHZ_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_BHZ_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_BHZ_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_BHZ_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_BHZ_1_t2.nii.gz'},
            'BraTS19_CBICA_BNR_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_BNR_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_BNR_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_BNR_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_CBICA_BNR_1_t2.nii.gz'},
            'BraTS19_TCIA01_180_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA01_180_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA01_180_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA01_180_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA01_180_1_t2.nii.gz'},
            'BraTS19_TCIA01_412_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA01_412_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA01_412_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA01_412_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA01_412_1_t2.nii.gz'},
            'BraTS19_TCIA01_448_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA01_448_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA01_448_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA01_448_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA01_448_1_t2.nii.gz'},
            'BraTS19_TCIA02_171_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_171_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_171_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_171_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_171_1_t2.nii.gz'},
            'BraTS19_TCIA02_198_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_198_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_198_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_198_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_198_1_t2.nii.gz'},
            'BraTS19_TCIA02_222_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_222_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_222_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_222_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_222_1_t2.nii.gz'},
            'BraTS19_TCIA02_290_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_290_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_290_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_290_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_290_1_t2.nii.gz'},
            'BraTS19_TCIA02_300_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_300_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_300_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_300_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_300_1_t2.nii.gz'},
            'BraTS19_TCIA02_331_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_331_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_331_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_331_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_331_1_t2.nii.gz'},
            'BraTS19_TCIA02_491_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_491_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_491_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_491_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA02_491_1_t2.nii.gz'},
            'BraTS19_TCIA03_121_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA03_121_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA03_121_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA03_121_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA03_121_1_t2.nii.gz'},
            'BraTS19_TCIA03_133_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA03_133_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA03_133_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA03_133_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA03_133_1_t2.nii.gz'},
            'BraTS19_TCIA03_338_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA03_338_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA03_338_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA03_338_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA03_338_1_t2.nii.gz'},
            'BraTS19_TCIA05_444_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA05_444_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA05_444_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA05_444_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA05_444_1_t2.nii.gz'},
            'BraTS19_TCIA06_165_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA06_165_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA06_165_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA06_165_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA06_165_1_t2.nii.gz'},
            'BraTS19_TCIA08_105_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA08_105_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA08_105_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA08_105_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA08_105_1_t2.nii.gz'},
            'BraTS19_TCIA08_205_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA08_205_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA08_205_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA08_205_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA08_205_1_t2.nii.gz'},
            'BraTS19_TCIA08_469_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA08_469_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA08_469_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA08_469_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TCIA08_469_1_t2.nii.gz'},
            'BraTS19_TMC_27374_1': {
                'flair': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TMC_27374_1_flair.nii.gz',
                't1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TMC_27374_1_t1.nii.gz',
                't1ce': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TMC_27374_1_t1ce.nii.gz',
                't2': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\imagesTr\\BraTS19_TMC_27374_1_t2.nii.gz'}},
        'label': {
            'BraTS19_2013_18_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_2013_18_1_seg.nii.gz',
            'BraTS19_2013_21_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_2013_21_1_seg.nii.gz',
            'BraTS19_2013_22_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_2013_22_1_seg.nii.gz',
            'BraTS19_2013_2_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_2013_2_1_seg.nii.gz',
            'BraTS19_2013_5_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_2013_5_1_seg.nii.gz',
            'BraTS19_2013_7_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_2013_7_1_seg.nii.gz',
            'BraTS19_CBICA_ABB_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_ABB_1_seg.nii.gz',
            'BraTS19_CBICA_ABE_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_ABE_1_seg.nii.gz',
            'BraTS19_CBICA_ALN_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_ALN_1_seg.nii.gz',
            'BraTS19_CBICA_AME_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_AME_1_seg.nii.gz',
            'BraTS19_CBICA_AOS_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_AOS_1_seg.nii.gz',
            'BraTS19_CBICA_AOZ_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_AOZ_1_seg.nii.gz',
            'BraTS19_CBICA_AQN_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_AQN_1_seg.nii.gz',
            'BraTS19_CBICA_AQT_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_AQT_1_seg.nii.gz',
            'BraTS19_CBICA_ARW_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_ARW_1_seg.nii.gz',
            'BraTS19_CBICA_ARZ_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_ARZ_1_seg.nii.gz',
            'BraTS19_CBICA_ASA_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_ASA_1_seg.nii.gz',
            'BraTS19_CBICA_ASK_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_ASK_1_seg.nii.gz',
            'BraTS19_CBICA_ASO_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_ASO_1_seg.nii.gz',
            'BraTS19_CBICA_ATN_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_ATN_1_seg.nii.gz',
            'BraTS19_CBICA_AUQ_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_AUQ_1_seg.nii.gz',
            'BraTS19_CBICA_AUW_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_AUW_1_seg.nii.gz',
            'BraTS19_CBICA_AWH_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_AWH_1_seg.nii.gz',
            'BraTS19_CBICA_AWX_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_AWX_1_seg.nii.gz',
            'BraTS19_CBICA_AXN_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_AXN_1_seg.nii.gz',
            'BraTS19_CBICA_AXO_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_AXO_1_seg.nii.gz',
            'BraTS19_CBICA_AZH_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_AZH_1_seg.nii.gz',
            'BraTS19_CBICA_BCL_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_BCL_1_seg.nii.gz',
            'BraTS19_CBICA_BGN_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_BGN_1_seg.nii.gz',
            'BraTS19_CBICA_BGX_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_BGX_1_seg.nii.gz',
            'BraTS19_CBICA_BHZ_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_BHZ_1_seg.nii.gz',
            'BraTS19_CBICA_BNR_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_CBICA_BNR_1_seg.nii.gz',
            'BraTS19_TCIA01_180_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_TCIA01_180_1_seg.nii.gz',
            'BraTS19_TCIA01_412_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_TCIA01_412_1_seg.nii.gz',
            'BraTS19_TCIA01_448_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_TCIA01_448_1_seg.nii.gz',
            'BraTS19_TCIA02_171_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_TCIA02_171_1_seg.nii.gz',
            'BraTS19_TCIA02_198_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_TCIA02_198_1_seg.nii.gz',
            'BraTS19_TCIA02_222_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_TCIA02_222_1_seg.nii.gz',
            'BraTS19_TCIA02_290_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_TCIA02_290_1_seg.nii.gz',
            'BraTS19_TCIA02_300_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_TCIA02_300_1_seg.nii.gz',
            'BraTS19_TCIA02_331_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_TCIA02_331_1_seg.nii.gz',
            'BraTS19_TCIA02_491_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_TCIA02_491_1_seg.nii.gz',
            'BraTS19_TCIA03_121_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_TCIA03_121_1_seg.nii.gz',
            'BraTS19_TCIA03_133_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_TCIA03_133_1_seg.nii.gz',
            'BraTS19_TCIA03_338_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_TCIA03_338_1_seg.nii.gz',
            'BraTS19_TCIA05_444_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_TCIA05_444_1_seg.nii.gz',
            'BraTS19_TCIA06_165_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_TCIA06_165_1_seg.nii.gz',
            'BraTS19_TCIA08_105_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_TCIA08_105_1_seg.nii.gz',
            'BraTS19_TCIA08_205_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_TCIA08_205_1_seg.nii.gz',
            'BraTS19_TCIA08_469_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_TCIA08_469_1_seg.nii.gz',
            'BraTS19_TMC_27374_1': 'C:\\Users\\melandur\\Desktop\\mo\\my_test\\labelsTr\\BraTS19_TMC_27374_1_seg.nii.gz'}}

    from main_conf import ConfigManager

    params = ConfigManager().params

    da = DataAugmentation(params)
    x = da.test(data_dicts)
    print(x)
    # print(x)
    #
    # for case in data_dicts['image']:
    #     print(case)
    #     x = tf.ConcatItemsD()
    #
    #     print(x)

    # print(x)
    # print('image shape:', np.shape(x['image']))
    # print('label shape:', np.shape(x['label']))
