import monai.transforms as tf
import numpy as np
import torch
from monai.transforms.compose import MapTransform
from monai.transforms.utils import generate_spatial_bounding_box
from skimage.transform import resize


class ToOneHot(MapTransform):
    def __init__(self, keys, label_classes: dict):
        super().__init__(self)
        self.keys = keys
        self.label_classes = label_classes

    def __call__(self, data):
        for key in self.keys:
            store = []
            for _, class_label in self.label_classes.items():
                label_mask = torch.zeros_like(data[key])
                label_mask[data[key] == class_label] = 1
                store.append(label_mask)
            data[key] = torch.vstack(store)
        return data


def get_transforms(config: dict, mode: str) -> tf.Compose:
    # if mode == 'test':
    #     keys = ['image']
    # else:
    keys = ['image', 'label']

    load_transforms = [
        tf.LoadImaged(keys=keys),
        tf.EnsureChannelFirstd(keys=keys),
        tf.ToTensord(keys='image'),
    ]

    # sample_transforms = [
    #     PreprocessAnisotropic(
    #         keys=keys,
    #         clip_values=config['dataset']['clip_values'],
    #         pixdim=config['dataset']['spacing'],
    #         normalize_values=config['dataset']['normalize_values'],
    #         model_mode=mode,
    #     ),
    #     tf.ToTensord(keys='image'),
    # ]

    if mode == 'train':

        spatial_transforms = [
            tf.CropForegroundd(keys=['image', 'label'], source_key='image', allow_smaller=False),
            tf.NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
            # tf.RandGaussianSmoothd(
            #     keys=['image'],
            #     sigma_x=(0.5, 1.15),
            #     sigma_y=(0.5, 1.15),
            #     sigma_z=(0.5, 1.15),
            #     prob=0.15,
            # ),
            # tf.RandScaleIntensityd(['image'], channel_wise=True, factors=0.1, prob=0.5),
            # tf.RandShiftIntensityd(['image'], channel_wise=True, offsets=0.1, prob=0.5),
            # tf.RandGaussianNoised(['image'], std=0.01, prob=0.15),
            # tf.NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
            tf.SpatialPadd(['image', 'label'], spatial_size=config['trainer']['patch_size']),
            tf.RandCropByLabelClassesd(
                keys=['image', 'label'],
                label_key='label',
                spatial_size=config['trainer']['patch_size'],
                num_classes=len(config['trainer']['label_classes']),
                num_samples=1,
                warn=False,
            ),
            tf.RandRotated(['image', 'label'], range_x=0.4, range_y=0.4, range_z=0.4, prob=0.2),
            tf.RandZoomd(
                keys=['image', 'label'],
                min_zoom=0.9,
                max_zoom=1.2,
                mode=('trilinear', 'nearest'),
                align_corners=(True, None),
                prob=0.5,
            ),
            tf.RandFlipd(['image', 'label'], spatial_axis=0, prob=0.5),
            tf.RandFlipd(['image', 'label'], spatial_axis=1, prob=0.5),
            tf.RandFlipd(['image', 'label'], spatial_axis=2, prob=0.5),

        ]
        other = [
            ToOneHot(keys=['label'], label_classes=config['trainer']['label_classes']),
            tf.CastToTyped(keys=['image', 'label'], dtype=(np.float32, np.uint8)),
            tf.EnsureTyped(keys=['image', 'label']),
        ]

        spatial_transforms = spatial_transforms + other

    elif mode == 'val':
        spatial_transforms = [
            tf.NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
            ToOneHot(keys=['label'], label_classes=config['trainer']['label_classes']),
            tf.CastToTyped(keys=['image', 'label'], dtype=(np.float32, np.uint8)),
            tf.EnsureTyped(keys=['image', 'label']),
        ]
    else:
        spatial_transforms = [
            tf.NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
            ToOneHot(keys=['label'], label_classes=config['trainer']['label_classes']),
            tf.CastToTyped(keys=['image'], dtype=(np.float32)),
            tf.EnsureTyped(keys=['image']),
        ]

    all_transforms = load_transforms + spatial_transforms
    return tf.Compose(all_transforms)


def recovery_prediction(prediction, shape, anisotrophy_flag):
    reshaped = np.zeros(shape, dtype=np.uint8)
    n_class = shape[0]
    if anisotrophy_flag:
        c, h, w = prediction.shape[:-1]
        d = shape[-1]
        reshaped_d = np.zeros((c, h, w, d), dtype=np.uint8)
        for class_ in range(1, n_class):
            mask = prediction[class_] == 1
            resized_d = resize(
                mask.astype(float),
                (h, w, d),
                order=0,
                mode='constant',
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            reshaped_d[class_][resized_d >= 0.5] = 1

        for class_ in range(1, n_class):
            for depth_ in range(d):
                mask = reshaped_d[class_, :, :, depth_] == 1
                resized_hw = resize(
                    mask.astype(float),
                    shape[1:-1],
                    order=1,
                    mode='edge',
                    cval=0,
                    clip=True,
                    anti_aliasing=False,
                )
                reshaped[class_, :, :, depth_][resized_hw >= 0.5] = 1
    else:
        for class_ in range(1, n_class):
            mask = prediction[class_] == 1
            resized = resize(
                mask.astype(float),
                shape[1:],
                order=1,
                mode='edge',
                cval=0,
                clip=True,
                anti_aliasing=False,
            )
            reshaped[class_][resized >= 0.5] = 1

    return reshaped


class PreprocessAnisotropic(MapTransform):
    """This transform class takes NNUNet's preprocessing method for reference."""

    def __init__(
        self,
        keys,
        clip_values,
        pixdim,
        normalize_values,
        model_mode,
    ) -> None:
        super().__init__(keys)
        self.keys = keys
        self.low = clip_values[0]
        self.high = clip_values[1]
        self.target_spacing = pixdim
        self.mean = normalize_values[0]
        self.std = normalize_values[1]
        self.training = False
        self.crop_foreground = tf.CropForegroundd(keys=['image', 'label'], source_key='image', allow_smaller=False)
        self.normalize_intensity = tf.NormalizeIntensity(nonzero=True, channel_wise=True)
        if model_mode in ['train']:
            self.training = True

    def calculate_new_shape(self, spacing, shape):
        spacing_ratio = np.array(spacing) / np.array(self.target_spacing)
        new_shape = (spacing_ratio * np.array(shape)).astype(int).tolist()
        return new_shape

    def check_anisotrophy(self, spacing):
        def check(spacing):
            return np.max(spacing) / np.min(spacing) >= 3

        return check(spacing) or check(self.target_spacing)

    def __call__(self, data):
        # load data
        d = dict(data)
        image = d['image']

        if 'label' in self.keys:
            label = d['label']
            label[label < 0] = 0

        # if self.training:
        cropped_data = self.crop_foreground({'image': image, 'label': label})
        image, label = cropped_data['image'], cropped_data['label']
        # else:
        #     d['original_shape'] = np.array(image.shape[1:])
        #     box_start, box_end = generate_spatial_bounding_box(image, allow_smaller=False)
        #     image = SpatialCrop(roi_start=box_start, roi_end=box_end)(image)
        #     d['bbox'] = np.vstack([box_start, box_end])
        #     d['crop_shape'] = np.array(image.shape[1:])

        image = image.numpy()
        if self.low != 0 or self.high != 0:
            image = np.clip(image, self.low, self.high)
            image = (image - self.mean) / self.std
        else:
            image = self.normalize_intensity(image.copy())

        d['image'] = image

        if 'label' in self.keys:
            d['label'] = label

        return d
