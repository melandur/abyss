import json
import os

import numpy as np
import SimpleITK as sitk
import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.transforms import AsDiscrete
from sliding_window import sliding_window_inference
from transforms import get_transforms

from abyss.config import ConfigFile
from abyss.training.create_network import get_network

dataset_path = '/home/melandur/Downloads/ucsf_corr/ucsf_images'
dst = '/home/melandur/Downloads/ucsf_corr/ucsf_seg/ucsf_test'
# checkpoint_path = '/home/melandur/Downloads/aby/train/1_results/best-epoch=930-loss_val=0.26.ckpt'
checkpoint_path = '/home/melandur/Downloads/aby/train/1_results/best-epoch=895-loss_val=0.23.ckpt'
datalist_file = '/home/melandur/Downloads/ucsf_corr/inference_dataset.json'

with open(datalist_file, 'r') as path:
    data_dict = json.load(path)

datalist = data_dict['inference']
for subject in datalist:
    subject['image'] = [os.path.join(dataset_path, subject['name'], image) for image in subject['image']]

config_file = ConfigFile()
config = config_file.get_config()

net = get_network(config)
checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['state_dict']
for key in list(state_dict.keys()):
    if 'net.' in key:
        new_key = key.replace('net.', '')
        state_dict[new_key] = state_dict.pop(key)
    if 'criterion.' in key:
        state_dict.pop(key)

net.load_state_dict(state_dict)
net = net.cuda()
net.eval()

transform = get_transforms(config, 'inference')
dataset = CacheDataset(
    data=datalist,
    transform=transform,
    num_workers=config['training']['num_workers'],
    cache_rate=config['training']['cache_rate'],
    copy_cache=False,
)

data_loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=config['training']['num_workers'],
    drop_last=False,
)

for batch in data_loader:
    with torch.amp.autocast(device_type='cuda'):
        print(batch['name'][0])
        name = batch['name'][0]
        data = batch['image'].as_tensor()
        pred = sliding_window_inference(data, (128, 128, 128), net)

        counts = 1.0
        for dims in [[2], [3], [4], (2, 3), (2, 4), (3, 4), (2, 3, 4)]:
            flip_inputs = torch.flip(data, dims=dims)
            flip_pred = torch.flip(sliding_window_inference(flip_inputs, (128, 128, 128), net).cpu(), dims=dims)
            del flip_inputs
            pred += flip_pred
            del flip_pred
            counts += 1.0
        pred = pred / counts

        # pred = pred[:, 1:, ...]  # remove background
        pred = torch.nn.functional.sigmoid(pred)
        post_pred = AsDiscrete(threshold=0.5)
        pred = post_pred(decollate_batch(pred)[0])

        mask = torch.zeros_like(pred[0])
        x = pred[1] - pred[0]
        mask[x == 1] = 1
        x = pred[0] - pred[2]
        mask[x == 1] = 3
        mask[pred[2] == 1] = 2

        mask = mask.permute(2, 1, 0)
        mask_arr = mask.numpy()
        img = sitk.GetImageFromArray(mask_arr)
        img.SetOrigin(np.concatenate([t.numpy() for t in batch['origin']]))
        img.SetDirection(np.concatenate([t.numpy() for t in batch['orientation']]))
        export_path = os.path.join(dst, name)
        os.makedirs(export_path, exist_ok=True)
        sitk.WriteImage(img, os.path.join(export_path, f'{name}_pred.nii.gz'))
