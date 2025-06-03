import monai.transforms as tf
import numpy as np
import SimpleITK as sitk
import torch
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import BGContrast, ContrastTransform
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from monai.data import MetaTensor
from monai.transforms import LoadImage
from monai.transforms.transform import MapTransform


class ToOneHot(MapTransform):
    def __init__(self, keys: list, label_classes: dict):
        super().__init__(self)
        self.keys = keys
        self.label_classes = label_classes

    def __call__(self, data):
        for key in self.keys:
            store = []

            for _, class_labels in self.label_classes.items():
                label_mask = torch.zeros_like(data[key])
                for class_label in class_labels:
                    label_mask[data[key] == class_label] = 1
                store.append(label_mask)
            data[key] = torch.vstack(store)
        return data


class CustomLoadImaged(MapTransform):
    def __init__(
        self,
        keys,
        image_key='image',
        label_key='label',
        meta_key_postfix='meta_dict',
        allow_missing_keys=False,
        inference=False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.image_key = image_key
        self.label_key = label_key
        self.meta_key_postfix = meta_key_postfix
        self.inference = inference
        self.loader = LoadImage(image_only=True)

    def __call__(self, data):
        d = dict(data)

        if self.inference:
            img = sitk.ReadImage(d['image'][0])
            d['origin'] = img.GetOrigin()
            d['orientation'] = img.GetDirection()
            d['spacing'] = img.GetSpacing()

        if 'image' in d:
            images = [self.loader(image_path) for image_path in d['image']]
            stacked_images = np.stack(images, axis=0)
            d['image'] = MetaTensor(stacked_images)

        if 'label' in d:
            labels = [self.loader(label_path) for label_path in d['label']]
            stacked_labels = np.stack(labels, axis=0)
            d['label'] = MetaTensor(stacked_labels)
        return d


class SpatialTrans(MapTransform):
    def __init__(self, keys=['image', 'label'], patch_size=(128, 128, 128)):
        super().__init__(keys)
        self.patch_size = patch_size

    def __call__(self, data):
        d = dict(data)
        transform = SpatialTransform(
            patch_size=self.patch_size,
            patch_center_dist_from_border=0,
            random_crop=False,
            p_elastic_deform=0,
            p_rotation=0.2,
            rotation=(-np.pi, np.pi),  # isotropic rotation
            p_scaling=0.2,
            scaling=(0.6, 1.5),
            p_synchronize_scaling_across_axes=1,
            bg_style_seg_sampling=False,
        )
        data_dict = {'image': d['image'], 'segmentation': d['label']}
        output_dict = transform(**data_dict)
        d['image'] = output_dict['image']
        d['label'] = output_dict['segmentation']
        return d


class GaussianNoiseTrans(MapTransform, tf.RandomizableTransform):
    def __init__(self, keys=['image'], prob=0.1):
        super().__init__(keys)
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        if self.R.rand() < self.prob:
            transform = GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_channel=1, synchronize_channels=True)
            data_dict = {'image': d['image']}
            output_dict = transform(**data_dict)
            d['image'] = output_dict['image']
        return d


class GaussianBlurTrans(MapTransform, tf.RandomizableTransform):
    def __init__(self, keys=['image'], prob=0.2):
        super().__init__(keys)
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        if self.R.rand() < self.prob:
            transform = GaussianBlurTransform(
                blur_sigma=(0.5, 1.0),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5,
                benchmark=True,
            )
            data_dict = {'image': d['image']}
            output_dict = transform(**data_dict)
            d['image'] = output_dict['image']
        return d


class MultiplicativeBrightnessTrans(MapTransform, tf.RandomizableTransform):
    def __init__(self, keys=['image'], prob=0.15):
        super().__init__(keys)
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        if self.R.rand() < self.prob:
            transform = MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)), synchronize_channels=False, p_per_channel=1
            )
            data_dict = {'image': d['image']}
            output_dict = transform(**data_dict)
            d['image'] = output_dict['image']
        return d


class ContrastTrans(MapTransform, tf.RandomizableTransform):
    def __init__(self, keys=['image'], prob=0.15):
        super().__init__(keys)
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        if self.R.rand() < self.prob:
            transform = ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1,
            )
            data_dict = {'image': d['image']}
            output_dict = transform(**data_dict)
            d['image'] = output_dict['image']
        return d


class SimulateLowResolutionTrans(MapTransform, tf.RandomizableTransform):
    def __init__(self, keys=['image'], prob=0.25):
        super().__init__(keys)
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        if self.R.rand() < self.prob:
            transform = SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=None,
                allowed_channels=None,
                p_per_channel=0.5,
            )
            data_dict = {'image': d['image']}
            output_dict = transform(**data_dict)
            d['image'] = output_dict['image']
        return d


class GammaTrans(MapTransform, tf.RandomizableTransform):
    def __init__(self, keys=['image'], prob=0.1):
        super().__init__(keys)
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        if self.R.rand() < self.prob:
            transform = GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1,
            )
            data_dict = {'image': d['image']}
            output_dict = transform(**data_dict)
            d['image'] = output_dict['image']
        return d


class MirrorTrans(MapTransform):
    def __init__(self, keys=['image', 'label']):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        transform = MirrorTransform(allowed_axes=(0, 1, 2))  # assuming 3D data
        data_dict = {'image': d['image'], 'segmentation': d['label']}
        output_dict = transform(**data_dict)
        d['image'] = output_dict['image']
        d['label'] = output_dict['segmentation']
        return d


def get_transforms(config: dict, mode: str) -> tf.Compose:
    keys = ['image', 'label']

    if mode == 'inference':
        keys = ['image']

    if mode == 'train':
        load_transforms = [
            CustomLoadImaged(keys=keys),
            tf.Orientationd(keys=['image', 'label'], axcodes='RAS'),
            tf.ToTensord(keys='image'),
        ]

        spatial_transforms = [
            tf.NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
            tf.CropForegroundd(
                keys=['image', 'label'], margin=[100, 100, 100], source_key='label', allow_smaller=False
            ),
            SpatialTrans(patch_size=config['trainer']['patch_size']),
            GaussianNoiseTrans(prob=0.1),
            GaussianBlurTrans(prob=0.2),
            MultiplicativeBrightnessTrans(prob=0.15),
            ContrastTrans(prob=0.15),
            SimulateLowResolutionTrans(prob=0.25),
            GammaTrans(prob=0.1),
            GammaTrans(prob=0.3),
            MirrorTrans(),
            ToOneHot(keys=['label'], label_classes=config['trainer']['label_classes']),
            tf.CastToTyped(keys=['image', 'label'], dtype=(np.float32, np.uint8)),
            tf.EnsureTyped(keys=['image', 'label']),
        ]

    elif mode == 'val':
        load_transforms = [
            CustomLoadImaged(keys=keys),
            tf.Orientationd(keys=['image', 'label'], axcodes='RAS'),
            tf.ToTensord(keys='image'),
        ]

        spatial_transforms = [
            tf.NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
            SpatialTrans(patch_size=config['trainer']['patch_size']),
            MirrorTrans(),
            ToOneHot(keys=['label'], label_classes=config['trainer']['label_classes']),
            tf.CastToTyped(keys=['image', 'label'], dtype=(np.float32, np.uint8)),
            tf.EnsureTyped(keys=['image', 'label']),
        ]
    else:
        load_transforms = [
            CustomLoadImaged(keys=keys, inference=True),
            tf.ToTensord(keys='image'),
        ]

        spatial_transforms = [
            tf.NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
            tf.CastToTyped(keys=['image'], dtype=np.float32),
            tf.EnsureTyped(keys=['image']),
        ]

    all_transforms = load_transforms + spatial_transforms
    return tf.Compose(all_transforms)
