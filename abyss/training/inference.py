import os

import torch
import torch.distributed as dist
from create_dataset import get_loader
from create_network import get_network
from inferrer import DynUNetInferrer
from monai.inferers import SlidingWindowInferer
from torch.nn.parallel import DistributedDataParallel


def inference(config) -> None:

    inference_path = config['project']['inference_path']
    os.makedirs(inference_path, exist_ok=True)

    if config['training']['multi_gpu']:
        dist.init_process_group(backend='nccl', init_method='env://')
        device = torch.device(f'cuda:{config["training"]["local_rank"]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cuda')

    properties, test_loader = get_loader(config, mode='test')

    net = get_network(config)
    net = net.to(device)

    if config['training']['multi_gpu']:
        net = DistributedDataParallel(module=net, device_ids=[device], find_unused_parameters=True)

    net.eval()

    inferrer = DynUNetInferrer(
        device=device,
        val_data_loader=test_loader,
        network=net,
        output_dir=config['project']['inference_path'],
        num_classes=len(properties['labels']),
        inferer=SlidingWindowInferer(
            roi_size=config['trainer']['patch_size'],
            sw_batch_size=3,
            overlap=0.5,
            mode='gaussian',
        ),
        amp=config['training']['amp'],
        tta_val=config['trainer']['tta'],
    )

    inferrer.run()
