import torch
from monai.losses import DiceCELoss, DiceLoss
from torch.nn import functional as F


def get_configured_optimizer(params, parameters):
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
            lr=optimizer_params['Adam']['learning_rate'],
            weight_decay=optimizer_params['Adam']['weight_decay'],
        )
    raise ValueError('Invalid optimizer settings -> conf.py -> training -> optimizers -> ')


def apply_criterion(params, output, ground_truth):
    """Calculate loss according to criterion"""
    loss = torch.tensor([0], dtype=torch.float32)
    for criterion in params['training']['criterion']:
        if 'mse' == criterion:
            loss += F.mse_loss(output, ground_truth)
        if 'dice' == criterion:
            dice_loss = DiceLoss()
            loss += dice_loss(output, ground_truth)  # TODO: Not tested
        if 'cross_entropy' == criterion:
            loss += F.cross_entropy(output, ground_truth)
        if 'cross_entropy_dice' == criterion:
            dice_ce_loss = DiceCELoss()
            loss += dice_ce_loss(output, ground_truth)  # TODO: Not tested
    return loss
