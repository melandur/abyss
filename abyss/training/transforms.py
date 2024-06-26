import SimpleITK as sitk
import monai.transforms as tf
import numpy as np
import torch

import torch.nn.functional as F
from monai.transforms import MapTransform, LoadImage, RandAdjustContrast
from monai.data import MetaTensor
from skimage.transform import resize


class ToOneHot(MapTransform):
    def __init__(self, keys, label_classes: dict):
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
    def __init__(self, keys, image_key='image', label_key='label', meta_key_postfix='meta_dict',
                 allow_missing_keys=False, inference=False):
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
            # Load and stack images
        # images = [self.loader(image_path) for image_path in d[self.image_key]]
        # stacked_images = np.stack(images, axis=0)
        # d[self.image_key] = MetaTensor(stacked_images)
        # Load and stack labels
        if 'label' in d:
            labels = [self.loader(label_path) for label_path in d['label']]
            stacked_labels = np.stack(labels, axis=0)
            d['label'] = MetaTensor(stacked_labels)
        # labels = [self.loader(label_path) for label_path in d[self.label_key]]
        # stacked_labels = np.stack(labels, axis=0)
        # d[self.label_key] = MetaTensor(stacked_labels)

        return d


class MultiplicativeBrightnessTransform(MapTransform, tf.RandomizableTransform):
    def __init__(self, keys, multiplier_range=(0.75, 1.25), synchronize_channels=False, p_per_channel=1.0, prob=0.1):
        super().__init__(keys)
        self.keys = keys
        self.multiplier_range = multiplier_range
        self.synchronize_channels = synchronize_channels
        self.p_per_channel = p_per_channel
        self.prob = prob

    def randomize(self):
        self.factor = self.R.uniform(self.multiplier_range[0], self.multiplier_range[1])

    def __call__(self, data):
        d = dict(data)
        self.randomize()
        for key in self.keys:
            if self.synchronize_channels:
                d[key] = d[key] * self.factor
            else:
                for c in range(d[key].shape[0]):  # assuming the first dimension is the channel
                    zero_mask = d[key][c] == 0
                    if self.R.uniform() < self.p_per_channel:
                        d[key][c] = d[key][c] * self.factor
                    d[key][c][zero_mask] = 0
        return d


class GaussianNoiseTransform(MapTransform, tf.RandomizableTransform):
    def __init__(self, keys, noise_variance=(0, 0.1), p_per_channel=1.0, prob=0.1):
        super().__init__(keys)
        self.keys = keys
        self.noise_variance = noise_variance
        self.p_per_channel = p_per_channel
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        self.noise_std = np.sqrt(self.R.uniform(self.noise_variance[0], self.noise_variance[1]))
        for key in self.keys:
            for c in range(d[key].shape[0]):  # assuming the first dimension is the channel
                zero_mask = d[key][c] == 0
                if self.R.uniform() < self.p_per_channel:
                    noise = self.R.normal(0, self.noise_std, size=d[key][c].shape)
                    d[key][c] += torch.from_numpy(noise)
                d[key][c][zero_mask] = 0
        return d


class ContrastTransform(MapTransform, tf.RandomizableTransform):
    def __init__(self, keys, contrast_range=(0.75, 1.25), preserve_range=True, p_per_channel=1.0, prob=0.1):
        super().__init__(keys)
        self.keys = keys
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.p_per_channel = p_per_channel
        self.transform = RandAdjustContrast(prob=1.0, gamma=self.contrast_range)
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            for c in range(d[key].shape[0]):  # assuming channel-first format
                if np.random.rand() < self.p_per_channel:
                    zero_mask = d[key][c] == 0
                    if self.preserve_range:
                        min_val, max_val = d[key][c].min(), d[key][c].max()
                        d[key][c] = self.transform(d[key][c])
                        d[key][c] = torch.clip(d[key][c], min_val, max_val)
                    else:
                        d[key][c] = self.transform(d[key][c])
                    d[key][c][zero_mask] = 0
        return d


class SimulateLowResolutionTransform(MapTransform, tf.RandomizableTransform):
    def __init__(self, keys, scale=(0.5, 1), p_per_channel=0.5, prob=0.1):
        super().__init__(keys)
        self.keys = keys
        self.scale = scale
        self.p_per_channel = p_per_channel
        self.prob = prob
        self.upmodes = {
            1: 'linear',
            2: 'bilinear',
            3: 'trilinear'
        }

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            for c in range(d[key].shape[0]):
                if np.random.rand() < self.p_per_channel:
                    zero_mask = d[key][c] == 0
                    new_shape = [round(i * np.random.uniform(*self.scale)) for i in d[key][c].shape]
                    downsampled = F.interpolate(d[key][c][None, None], new_shape, mode=self.upmodes[d[key][c].ndim])
                    d[key][c] = F.interpolate(downsampled, d[key][c].shape, mode=self.upmodes[d[key][c].ndim])[0, 0]
                    d[key][c][zero_mask] = 0
        return d


class GammaTransform(tf.RandomizableTransform):
    def __init__(self, keys, gamma_range=(0.7, 1.5), prob=0.1):
        super().__init__()
        self.keys = keys
        self.gamma_range = gamma_range
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            gamma = self.R.uniform(low=self.gamma_range[0], high=self.gamma_range[1])
            for c in range(img.shape[0]):
                img[c] = self.apply_gamma(img[c], gamma)
            d[key] = img
        return d

    def apply_gamma(self, img, gamma):
        img_min = torch.min(img)
        img_max = torch.max(img)
        if img_min == img_max:  # Handle the case where img_min == img_max to avoid division by zero
            return img  # No change if the image has no variation
        zero_mask = img == 0  # Create a mask to keep zeros as zeros
        img_normalized = (img - img_min) / (img_max - img_min)  # Normalize the image to the [0, 1] range
        img_gamma_corrected = img_normalized ** gamma  # Apply gamma correction to non-zero values
        img_scaled = img_gamma_corrected * (img_max - img_min) + img_min          # Scale back to the original range
        img_scaled[zero_mask] = 0  # Restore the zeros
        return img_scaled


def get_transforms(config: dict, mode: str) -> tf.Compose:
    if mode == 'inference':
        keys = ['image']
    else:
        keys = ['image', 'label']

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

        load_transforms = [
            CustomLoadImaged(keys=keys),
            # tf.EnsureChannelFirstd(keys=keys),
            tf.ToTensord(keys='image'),
        ]

        spatial_transforms = [
            tf.CropForegroundd(keys=['image', 'label'], source_key='image', allow_smaller=False),
            tf.NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
            tf.SpatialPadd(['image', 'label'], spatial_size=config['trainer']['patch_size']),
            tf.RandFlipd(['image', 'label'], spatial_axis=0, prob=0.5),
            tf.RandFlipd(['image', 'label'], spatial_axis=1, prob=0.5),
            tf.RandFlipd(['image', 'label'], spatial_axis=2, prob=0.5),
            tf.RandCropByLabelClassesd(
                keys=['image', 'label'],
                label_key='label',
                image_key='image',
                ratios=[0.01, 1.0, 1.0, 1.0],
                spatial_size=config['trainer']['patch_size'],
                num_classes=len(config['trainer']['label_classes']),
                num_samples=1,
                allow_smaller=False,
                warn=False,
            ),
            tf.RandRotated(
                keys=['image', 'label'],
                mode=('bilinear', 'nearest'),
                align_corners=(True, None),
                padding_mode=('constant', 'constant'),
                range_x=3.14,
                range_y=3.14,
                range_z=3.14,
                prob=1.0,
            ),
            tf.RandZoomd(
                keys=['image', 'label'],
                min_zoom=0.8,
                max_zoom=1.3,
                mode=('bilinear', 'nearest'),
                align_corners=(True, None),
                padding_mode=('constant', 'constant'),
                prob=1.0,
            ),
            GaussianNoiseTransform(
                keys=['image'],
                noise_variance=(0, 0.1),
                p_per_channel=1.0,
                prob=0.1,
            ),
            tf.RandGaussianSmoothd(
                keys=['image'],
                sigma_x=(0.5, 1.),
                sigma_y=(0.5, 1.),
                sigma_z=(0.5, 1.),
                prob=0.2,
            ),
            MultiplicativeBrightnessTransform(
                keys=['image'],
                multiplier_range=(0.75, 1.25),
                synchronize_channels=False,
                p_per_channel=1.0,
                prob=0.15,
            ),
            ContrastTransform(
                keys=['image'],
                contrast_range=(0.75, 1.25),
                preserve_range=True,
                p_per_channel=1.0,
                prob=0.15,
            ),
            SimulateLowResolutionTransform(
                keys=['image'],
                scale=(0.5, 1.0),
                p_per_channel=0.5,
                prob=0.25,
            ),
            GammaTransform(
                keys=['image'],
                gamma_range=(0.5, 1.2),
                prob=0.3,
            ),
        ]
        other = [
            ToOneHot(keys=['label'], label_classes=config['trainer']['label_classes']),
            tf.CastToTyped(keys=['image', 'label'], dtype=(np.float32, np.uint8)),
            tf.EnsureTyped(keys=['image', 'label']),
        ]

        spatial_transforms = spatial_transforms + other

    elif mode == 'val':
        load_transforms = [
            CustomLoadImaged(keys=keys),
            # tf.EnsureChannelFirstd(keys=keys),
            tf.ToTensord(keys='image'),
        ]

        spatial_transforms = [
            tf.CropForegroundd(keys=['image', 'label'], source_key='image', allow_smaller=False),
            tf.NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
            ToOneHot(keys=['label'], label_classes=config['trainer']['label_classes']),
            tf.CastToTyped(keys=['image', 'label'], dtype=(np.float32, np.uint8)),
            tf.EnsureTyped(keys=['image', 'label']),
        ]
    else:
        load_transforms = [
            CustomLoadImaged(keys=keys, inference=True),
            # tf.EnsureChannelFirstd(keys=keys),
            tf.ToTensord(keys='image'),
        ]

        spatial_transforms = [
            tf.NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
            tf.CastToTyped(keys=['image'], dtype=np.float32),
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
