import logging

import ignite.distributed as idist
import torch
import torch.distributed as dist
from ignite.contrib.handlers import create_lr_scheduler_with_warmup
from ignite.engine import Events
from ignite.handlers import EarlyStopping
from monai.handlers import CheckpointSaver, LrScheduleHandler, MeanDice, StatsHandler, ValidationHandler, from_engine
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.utils import set_determinism
from torch.nn.parallel import DistributedDataParallel

from abyss.training.create_dataset import get_loader
from abyss.training.create_network import get_network
from abyss.training.evaluator import DynUNetEvaluator
from abyss.training.trainer import DynUNetTrainer


def validation(config: dict) -> None:

    if config['training']['multi_gpu']:
        dist.init_process_group(backend='nccl', init_method='env://')
        device = torch.device(f'cuda:{config["training"]["local_rank"]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cuda')

    val_loader = get_loader(config, mode='val')
    net = get_network(config)
    net = net.to(device)

    if config['training']['multi_gpu']:
        net = DistributedDataParallel(module=net, device_ids=[device])

    net.eval()
    num_classes = 4
    evaluator = DynUNetEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        num_classes=num_classes,
        inferer=SlidingWindowInferer(
            roi_size=config['trainer']['patch_size'],
            sw_batch_size=4,
            overlap=0.5,
            mode='gaussian',
        ),
        postprocessing=None,
        key_val_metric={
            'val_mean_dice': MeanDice(
                include_background=False,
                output_transform=from_engine(['pred', 'label']),
            )
        },
        additional_metrics=None,
        amp=config['training']['amp'],
        tta_val=config['trainer']['tta'],
    )

    evaluator.run()
    if config['training']['local_rank'] == 0:
        print(evaluator.state.metrics)
        results = evaluator.state.metric_details['val_mean_dice']
        if num_classes > 2:
            for i in range(num_classes - 1):
                print('mean dice for label {} is {}'.format(i + 1, results[:, i].mean()))

    if config['training']['multi_gpu']:
        dist.destroy_process_group()


def train(config: dict) -> None:

    if config['training']['deterministic']:
        set_determinism(seed=config['training']['seed'])
        if config['training']['local_rank'] == 0:
            logging.info('Set deterministic training')

    if config['training']['multi_gpu']:
        dist.init_process_group(backend='nccl', init_method='env://')
        device = torch.device(f'cuda:{config["training"]["local_rank"]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cuda')

    val_loader = get_loader(config, mode='val')
    train_loader = get_loader(config, mode='train')

    # produce the network
    net = get_network(config)

    if config['training']['compile']:
        net = torch.compile(net)  # todo: check this

    net = net.to(device)

    if config['training']['multi_gpu']:
        net = DistributedDataParallel(module=net, device_ids=[device])

    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=config['training']['learning_rate'],
        momentum=0.99,
        weight_decay=3e-5,
        nesterov=True,
    )

    max_epochs = config['training']['max_epochs']
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / max_epochs) ** 0.9)

    if config['training']['warmup']['active']:
        scheduler = create_lr_scheduler_with_warmup(
            scheduler,
            warmup_start_value=0.0,
            warmup_end_value=config['training']['learning_rate'],
            warmup_duration=config['training']['warmup']['epochs'],
        )

    # produce evaluator
    val_handlers = (
        [
            StatsHandler(output_transform=lambda x: None),
            CheckpointSaver(
                save_dir=config['project']['results_path'],
                save_dict={'net': net},
                save_key_metric=True,
            ),
        ]
        if idist.get_rank() == 0
        else None
    )

    evaluator = DynUNetEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        num_classes=4,
        inferer=SlidingWindowInferer(
            roi_size=config['trainer']['patch_size'],
            sw_batch_size=3,
            overlap=0.5,
            mode='gaussian',
        ),
        postprocessing=None,
        key_val_metric={
            'val_mean_dice': MeanDice(
                include_background=False,
                output_transform=from_engine(['pred', 'label']),
            )
        },
        val_handlers=val_handlers,
        amp=config['training']['amp'],
        tta_val=config['trainer']['tta'],
    )

    # produce trainer
    loss = DiceCELoss(to_onehot_y=True, softmax=True, batch=True)
    train_handlers = [
        ValidationHandler(validator=evaluator, interval=config['training']['val_interval'], epoch_level=True)
    ]

    if config['trainer']['lr_decay']:
        train_handlers += [LrScheduleHandler(lr_scheduler=scheduler, print_lr=True)]

    if idist.get_rank() == 0:
        train_handlers += [StatsHandler(tag_name='train_loss', output_transform=from_engine(['loss'], first=True))]

    trainer = DynUNetTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=optimizer,
        loss_function=loss,
        inferer=SimpleInferer(),
        postprocessing=None,
        key_train_metric=None,
        train_handlers=train_handlers,
        amp=config['training']['amp'],
    )

    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    if config['training']['early_stop']['active']:
        early_stop = EarlyStopping(
            patience=config['training']['early_stop']['patience'],
            score_function=lambda engine: -engine.state.metrics['val_mean_dice'],
            trainer=trainer,
        )
        evaluator.add_event_handler(Events.COMPLETED, early_stop)

    if config['training']['local_rank'] > 0:
        evaluator.logger.setLevel(logging.WARNING)
        trainer.logger.setLevel(logging.WARNING)

    trainer.run()
    if config['training']['multi_gpu']:
        dist.destroy_process_group()
