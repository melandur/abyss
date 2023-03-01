import torch
from monai.losses import DiceCELoss, DiceLoss
from torch.nn import functional as F


def get_optimizer(params, parameters):
    """Returns configured optimizer accordingly to the config file"""
    optimizer_params = params['training']['optimizers']
    if optimizer_params['Adam']['active']:
        return torch.optim.Adam(
            params=parameters(),
            lr=optimizer_params['Adam']['learning_rate'],
            betas=optimizer_params['Adam']['betas'],
            weight_decay=optimizer_params['Adam']['weight_decay'],
            eps=optimizer_params['Adam']['eps'],
            amsgrad=optimizer_params['Adam']['amsgrad'],
        )
    if optimizer_params['SGD']['active']:
        return torch.optim.SGD(
            params=parameters(),
            lr=optimizer_params['SGD']['learning_rate'],
            momentum=optimizer_params['SGD']['momentum'],
            weight_decay=optimizer_params['Adam']['weight_decay'],
            nesterov=optimizer_params['SGD']['nesterov'],
        )
    raise ValueError('Invalid optimizer settings -> conf.py -> training -> optimizers')


def apply_criterion(params, output, ground_truth):
    """Calculate loss according to criterion"""
    criterion = params['training']['criterion']
    if 'mse' == criterion:
        return F.mse_loss(output, ground_truth)
    if 'dice' == criterion:
        dice_loss = DiceLoss(include_background=False, to_onehot_y=True, softmax=True)
        return dice_loss(output, ground_truth)
    if 'cross_entropy' == criterion:
        return F.cross_entropy(output, ground_truth)
    if 'cross_entropy_dice' == criterion:
        dice_ce_loss = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        return dice_ce_loss(output, ground_truth)
    raise ValueError('Invalid criterion settings -> conf.py -> training -> criterion')
