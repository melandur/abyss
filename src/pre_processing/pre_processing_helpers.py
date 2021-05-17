import numpy as np
from loguru import logger as log
from monai import transforms as tf


class ConcatenateImages(tf.MapTransform):
    """
    Stack images on new channel which is added on the first dimensional position.
    MONAI expects a simple dict. But we like our dicts nested, here we go
    """

    def __init__(self, keys, case_name=0, dim=0, add_channel=True):
        super().__init__(keys)
        self.case_name = case_name
        self.dim = dim
        self.add_channel = add_channel

    def __call__(self, keys):
        d = dict(keys)

        def load_image_sub_dict(sub_dict, keys):
            """Load image and add channel to the first dimension"""
            if self.add_channel:
                load_image = tf.Compose([
                    tf.LoadImaged(keys=keys, image_only=True),
                    tf.AddChanneld(keys=keys)])  # add channel on first dimension
            else:
                load_image = tf.Compose([tf.LoadImaged(keys=keys, image_only=True)])
            return load_image(sub_dict)

        def concatenate_images_sub_dict(sub_dict, keys):
            """Concatenate all images of one case/sub dictionary and assigns the result to the concat_output key"""
            concatenate_images = tf.Compose([tf.ConcatItemsd(keys=keys, name='concat_output', dim=self.dim)])
            return concatenate_images(sub_dict)

        tmp_path_sub_dict = {}
        for image_name in d['image'][self.case_name].keys():  # create sub dict for a single case
            tmp_path_sub_dict[image_name] = d['image'][self.case_name][image_name]

        tmp_sub_images = {'concat_output': None}  # adds key to dict to store concat result later on
        for image_name in [*tmp_path_sub_dict.keys()]:  # load image from each image path in sub dict
            tmp = load_image_sub_dict(tmp_path_sub_dict, image_name)
            tmp_sub_images[image_name] = tmp[image_name]

        # assign concat results to input by overwriting the previous sub dicts
        tmp_image = concatenate_images_sub_dict(tmp_sub_images, [*tmp_path_sub_dict.keys()])
        log.debug(f'{self.case_name}, concatenated output dimension: {np.shape(tmp_image["concat_output"])}')
        d['image'][self.case_name] = tmp_image["concat_output"]
        del tmp_path_sub_dict
        del tmp_sub_images
        return d
