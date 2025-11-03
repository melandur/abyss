import json
import os

import numpy as np
import SimpleITK as sitk
import torch
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.transforms import AsDiscrete
from abyss.engine.sliding_window import sliding_window_inference
from abyss.transforms.transforms import get_segmentation_transforms
from abyss.models.create_network import get_network
from abyss.config import ConfigFile

dataset_path = '/home/melandur/Downloads/final_mets_test'
dst = '/home/melandur/Downloads/test'
# checkpoint_path = '/home/melandur/Downloads/aby/train/1_results/best-epoch=930-loss_val=0.26.ckpt'
checkpoint_path = '/home/melandur/Downloads/aby'
datalist_file = '/home/melandur/Downloads/inference_dataset.json'

with open(datalist_file, 'r') as path:
    data_dict = json.load(path)

datalist = data_dict['inference']
for subject in datalist:
    subject['image'] = [os.path.join(dataset_path, subject['name'], image) for image in subject['image']]

config_file = ConfigFile()
config = config_file.get_config()
patch_size = config['trainer']['patch_size']


def get_net(net, fold):
    torch.cuda.empty_cache()

    checkpoint = torch.load(
        os.path.join(checkpoint_path, f'mets_fold_{fold}', '1_results', 'best.ckpt'), weights_only=False
    )
    state_dict = checkpoint['state_dict']
    new_state_dict = {}

    for key in list(state_dict.keys()):
        if 'net.' in key:
            new_state_dict[key.replace('net.', '_orig_mod.')] = state_dict[key]

    net.load_state_dict(new_state_dict)
    net.cuda()
    net.eval()
    return net


transform = get_segmentation_transforms(config, 'inference')
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

net = get_network(config)

with torch.amp.autocast(device_type='cuda'):
    for batch in data_loader:
        print(batch['name'][0])
        merged_pred = None

        for fold in range(config['dataset']['total_folds']):
            print(f'--> fold {fold}')
            net = get_net(net, fold)

            name = batch['name'][0]
            data = batch['image'].as_tensor()
            pred = sliding_window_inference(data, patch_size, net).cpu()

            counts = 1.0
            for dims in [[2], [3], [4], (2, 3), (2, 4), (3, 4), (2, 3, 4)]:
                flip_inputs = torch.flip(data, dims=dims)
                flip_pred = torch.flip(sliding_window_inference(flip_inputs, patch_size, net).cpu(), dims=dims)
                del flip_inputs
                pred += flip_pred
                del flip_pred
                counts += 1.0
            pred = pred / counts

            if merged_pred is None:
                merged_pred = pred
            else:
                merged_pred += pred

        pred = merged_pred / config['dataset']['total_folds']
        # pred = pred[:, 1:, ...]  # remove background
        pred = torch.nn.functional.sigmoid(pred)
        post_pred = AsDiscrete(threshold=0.5)
        pred = post_pred(decollate_batch(pred)[0])

        mask = torch.zeros_like(pred[0])
        # x = pred[1] - pred[2]
        # mask[x == 1] = 3
        x = pred[0] - pred[1]
        mask[x == 1] = 2
        mask[pred[1] == 1] = 1

        mask = mask.permute(2, 1, 0)
        mask_arr = mask.numpy()
        img = sitk.GetImageFromArray(mask_arr)
        img.SetOrigin(np.concatenate([t.numpy() for t in batch['origin']]))
        img.SetDirection(np.concatenate([t.numpy() for t in batch['orientation']]))
        export_path = os.path.join(dst, name)
        os.makedirs(export_path, exist_ok=True)
        sitk.WriteImage(img, os.path.join(export_path, f'{name}_pred_aby.nii.gz'))
