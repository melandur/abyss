import copy

import numpy as np
import torch
import torchio as tio
from torchio.transforms.augmentation.random_transform import RandomTransform


class RandomChannelSkip(RandomTransform, tio.IntensityTransform):
    """Skip random channel of image by replacing it with certain fill value"""

    def __init__(self, num_channels, fill_value=0.0, prob=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.fill_value = fill_value
        self.probability = self.parse_probability(prob)

    def apply_transform(self, subject):
        """Transform image data of subject"""
        transformed = self.skip_channel(subject)
        return transformed

    def skip_channel(self, subject):
        """Overwrite values of random channel with certain fill value"""
        count_channels = self.get_images_dict(subject)['data'].num_channels
        channels = list(range(count_channels))
        subject_data = self.get_images_dict(subject)['data']
        image_data = self.get_images_dict(subject)['data'].data
        skip_channels = np.random.choice(channels, self.num_channels)
        for skip_channel in skip_channels:
            tmp_channel = torch.zeros(image_data[skip_channel].size(), dtype=torch.float32)
            image_data[skip_channel] = tmp_channel.new_full(tmp_channel.size(), self.fill_value)
        subject_data.set_data(image_data)
        return subject


class RandomChannelShuffle(RandomTransform, tio.IntensityTransform):
    """Shuffle Channels"""

    def __init__(self, prob=0.1, **kwargs):
        super().__init__(**kwargs)
        self.probability = self.parse_probability(prob)

    def apply_transform(self, subject):
        """Transform image data of subject"""
        transformed = self.shuffle_channels(subject)
        return transformed

    def shuffle_channels(self, subject):
        """Shuffle order of channels"""
        count_channels = self.get_images_dict(subject)['data'].num_channels
        channels = list(range(count_channels))
        subject_data = self.get_images_dict(subject)['data']
        image_data = self.get_images_dict(subject)['data'].data
        tmp_image_data = copy.deepcopy(image_data)
        np.random.shuffle(channels)
        for idx, channel in enumerate(channels):
            image_data[idx] = tmp_image_data[channel]
        subject_data.set_data(image_data)
        return subject
