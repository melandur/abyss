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
    def __init__(self, keys: list, label_classes: dict) -> None:
        super().__init__(self)
        self.keys = keys
        self.label_classes = label_classes

    def __call__(self, data) -> dict:
        for key in self.keys:
            store = []

            for _, class_labels in self.label_classes.items():
                label_mask = torch.zeros_like(data[key])
                for class_label in class_labels:
                    label_mask[data[key] == class_label] = 1
                store.append(label_mask)
            data[key] = torch.vstack(store)
        return data


class CustomLoadImagedClassification(MapTransform):
    def __init__(
        self,
        keys,
        image_key='image',
        label_key='label',
        meta_key_postfix='meta_dict',
        allow_missing_keys=False,
        inference=False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.image_key = image_key
        self.label_key = label_key
        self.meta_key_postfix = meta_key_postfix
        self.inference = inference
        self.loader = LoadImage(image_only=True)

    def __call__(self, data) -> dict:
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

        def label_path_to_int(label_path):
            with open(label_path, 'r') as f:
                label = np.array(f.read().strip().split(), dtype=np.float32)
            return label

        if 'label' in d:
            label = [label_path_to_int(label_path) for label_path in d['label']]
            d['label'] = MetaTensor(label)

        return d


class CustomLoadImagedSegmentation(MapTransform):
    def __init__(
        self,
        keys,
        image_key='image',
        label_key='label',
        meta_key_postfix='meta_dict',
        allow_missing_keys=False,
        inference=False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.image_key = image_key
        self.label_key = label_key
        self.meta_key_postfix = meta_key_postfix
        self.inference = inference
        self.loader = LoadImage(image_only=True)

    def __call__(self, data) -> dict:
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
    def __init__(self, keys, patch_size=(128, 128, 128)) -> None:
        super().__init__(keys)
        self.patch_size = patch_size

    def __call__(self, data) -> dict:
        d = dict(data)
        patch_center_dist = (
            min(p // 3 for p in self.patch_size) if isinstance(self.patch_size, (list, tuple)) else self.patch_size // 3
        )
        transform = SpatialTransform(
            patch_size=self.patch_size,
            patch_center_dist_from_border=patch_center_dist,
            random_crop=True,
            p_elastic_deform=0,
            p_rotation=0.2,
            rotation=(-np.pi / 6, np.pi / 6),
            p_scaling=0.2,
            scaling=(0.7, 1.4),
            p_synchronize_scaling_across_axes=1.0,
            bg_style_seg_sampling=True,
            border_mode_seg='border',
        )
        data_dict = {'image': d['image'], 'segmentation': d['label']}
        output_dict = transform(**data_dict)
        d['image'] = output_dict['image']
        d['label'] = output_dict['segmentation']
        return d


class GaussianNoiseTrans(MapTransform, tf.RandomizableTransform):
    def __init__(self, keys=['image'], prob=0.1) -> None:
        super().__init__(keys)
        self.prob = prob

    def __call__(self, data) -> dict:
        d = dict(data)
        if self.R.rand() < self.prob:
            transform = GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_channel=1, synchronize_channels=True)
            data_dict = {'image': d['image']}
            output_dict = transform(**data_dict)
            d['image'] = output_dict['image']
        return d


class GaussianBlurTrans(MapTransform, tf.RandomizableTransform):
    def __init__(self, keys=['image'], prob=0.2) -> None:
        super().__init__(keys)
        self.prob = prob

    def __call__(self, data) -> dict:
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
    def __init__(self, keys=['image'], prob=0.15) -> None:
        super().__init__(keys)
        self.prob = prob

    def __call__(self, data) -> dict:
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
    def __init__(self, keys=['image'], prob=0.15) -> None:
        super().__init__(keys)
        self.prob = prob

    def __call__(self, data) -> dict:
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
    def __init__(self, keys=['image'], prob=0.25) -> None:
        super().__init__(keys)
        self.prob = prob

    def __call__(self, data) -> dict:
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
    def __init__(self, keys=['image'], prob=0.1) -> None:
        super().__init__(keys)
        self.prob = prob

    def __call__(self, data) -> dict:
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
    def __init__(self, keys) -> None:
        super().__init__(keys)

    def __call__(self, data) -> dict:
        d = dict(data)
        transform = MirrorTransform(allowed_axes=(0, 1, 2))  # assuming 3D data
        data_dict = {'image': d['image'], 'segmentation': d['label']}
        output_dict = transform(**data_dict)
        d['image'] = output_dict['image']
        d['label'] = output_dict['segmentation']
        return d


def get_classification_transforms(config: dict, mode: str) -> tf.Compose:

    if mode == 'train':
        load_transforms = [
            CustomLoadImagedClassification(keys=['image', 'label']),
            tf.Orientationd(keys=['image'], axcodes='RAS'),
            tf.ToTensord(keys='image'),
        ]

        spatial_transforms = [
            tf.NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
            SpatialTrans(keys=['image'], patch_size=config['trainer']['patch_size']),
            GaussianNoiseTrans(prob=0.1),
            GaussianBlurTrans(prob=0.2),
            MultiplicativeBrightnessTrans(prob=0.15),
            ContrastTrans(prob=0.15),
            SimulateLowResolutionTrans(prob=0.25),
            GammaTrans(prob=0.1),
            GammaTrans(prob=0.3),
            MirrorTrans(keys=['image']),
            tf.CastToTyped(keys=['image'], dtype=(np.float32)),
            tf.EnsureTyped(keys=['image']),
        ]

    elif mode == 'val':
        load_transforms = [
            CustomLoadImagedClassification(keys=['image', 'label']),
            tf.Orientationd(keys=['image'], axcodes='RAS'),
            tf.ToTensord(keys='image'),
        ]

        spatial_transforms = [
            tf.NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
            SpatialTrans(keys=['image'], patch_size=config['trainer']['patch_size']),
            MirrorTrans(keys=['image']),
            tf.CastToTyped(keys=['image'], dtype=(np.float32)),
            tf.EnsureTyped(keys=['image']),
        ]

    else:
        load_transforms = [
            CustomLoadImagedClassification(keys=['image'], inference=True),
            tf.ToTensord(keys='image'),
        ]

        spatial_transforms = [
            tf.NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
            tf.CastToTyped(keys=['image'], dtype=np.float32),
            tf.EnsureTyped(keys=['image']),
        ]

    all_transforms = load_transforms + spatial_transforms
    return tf.Compose(all_transforms)


def get_segmentation_transforms(config: dict, mode: str) -> tf.Compose:

    if mode == 'train':
        load_transforms = [
            CustomLoadImagedSegmentation(keys=['image', 'label']),
            tf.Orientationd(keys=['image', 'label'], axcodes='RAS'),
            tf.ToTensord(keys='image'),
        ]

        patch_size = config['trainer']['patch_size']
        crop_margin = [max(p // 2, 20) for p in patch_size]
        target_spacing = config['dataset'].get('target_spacing', config['dataset']['spacing'])

        spatial_transforms = [
            tf.Spacingd(
                keys=['image', 'label'],
                pixdim=target_spacing,
                mode=['bilinear', 'nearest'],
                padding_mode='border',
            ),
            tf.NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
            tf.CropForegroundd(keys=['image', 'label'], margin=crop_margin, source_key='label', allow_smaller=False),
            SpatialTrans(keys=['image', 'label'], patch_size=config['trainer']['patch_size']),
            GaussianNoiseTrans(prob=0.1),
            GaussianBlurTrans(prob=0.2),
            MultiplicativeBrightnessTrans(prob=0.15),
            ContrastTrans(prob=0.15),
            SimulateLowResolutionTrans(prob=0.25),
            GammaTrans(prob=0.1),
            GammaTrans(prob=0.3),
            MirrorTrans(keys=['image', 'label']),
            ToOneHot(keys=['label'], label_classes=config['trainer']['label_classes']),
            tf.CastToTyped(keys=['image', 'label'], dtype=(np.float32, np.uint8)),
            tf.EnsureTyped(keys=['image', 'label']),
        ]

    elif mode == 'val':
        load_transforms = [
            CustomLoadImagedSegmentation(keys=['image', 'label']),
            tf.Orientationd(keys=['image', 'label'], axcodes='RAS'),
            tf.ToTensord(keys='image'),
        ]

        target_spacing = config['dataset'].get('target_spacing', config['dataset']['spacing'])
        patch_size = config['trainer']['patch_size']
        crop_margin = [max(p // 2, 20) for p in patch_size]

        spatial_transforms = [
            tf.Spacingd(
                keys=['image', 'label'],
                pixdim=target_spacing,
                mode=['bilinear', 'nearest'],
                padding_mode='border',
            ),
            tf.NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
            tf.CropForegroundd(keys=['image', 'label'], margin=crop_margin, source_key='label', allow_smaller=False),
            tf.CenterSpatialCropd(keys=['image', 'label'], roi_size=patch_size),
            ToOneHot(keys=['label'], label_classes=config['trainer']['label_classes']),
            tf.CastToTyped(keys=['image', 'label'], dtype=(np.float32, np.uint8)),
            tf.EnsureTyped(keys=['image', 'label']),
        ]
    else:
        load_transforms = [
            CustomLoadImagedSegmentation(keys=['image'], inference=True),
            tf.Orientationd(keys=['image'], axcodes='RAS'),
            tf.ToTensord(keys='image'),
        ]

        target_spacing = config['dataset'].get('target_spacing', config['dataset']['spacing'])

        spatial_transforms = [
            tf.Spacingd(
                keys=['image'],
                pixdim=target_spacing,
                mode='bilinear',
                padding_mode='border',
            ),
            tf.NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
            tf.CastToTyped(keys=['image'], dtype=np.float32),
            tf.EnsureTyped(keys=['image']),
        ]

    all_transforms = load_transforms + spatial_transforms
    return tf.Compose(all_transforms)
