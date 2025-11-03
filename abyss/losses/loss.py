"""
Loss functions for segmentation, following nnUNet v2 implementation style.

Includes Dice loss, CrossEntropyLoss, TopK loss, and their combinations.
"""

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks.

    Supports both sigmoid (multi-label) and softmax (multi-class) modes.
    Based on nnUNet v2 implementation.

    Args:
        apply_nonlin: Activation function to apply before computing dice ('sigmoid' or 'softmax')
        batch_dice: If True, compute dice over batch dimension
        smooth: Smoothing factor for numerical stability
        do_bg: If False, ignore background class (class 0)
    """

    def __init__(
        self,
        apply_nonlin: Union[str, None] = 'sigmoid',
        batch_dice: bool = True,
        smooth: float = 1e-5,
        do_bg: bool = True,
    ) -> None:
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.smooth = smooth
        self.do_bg = do_bg

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss.

        Args:
            x: Network output [B, C, D, H, W] or [B, C, H, W]
            y: Ground truth [B, C, D, H, W] or [B, C, H, W] (one-hot) or [B, D, H, W] (class indices)

        Returns:
            Dice loss value
        """
        shp_x = x.shape

        # Handle class indices format (convert to one-hot if needed)
        if len(y.shape) == len(shp_x) - 1:  # y is [B, D, H, W] or [B, H, W]
            y = y.long()
            y_onehot = torch.zeros_like(x)
            y_onehot.scatter_(1, y.unsqueeze(1), 1)
            y = y_onehot

        # Apply activation
        if self.apply_nonlin == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.apply_nonlin == 'softmax':
            x = F.softmax(x, dim=1)
        elif self.apply_nonlin is None:
            pass  # Assume already activated
        else:
            raise ValueError(f'Unknown activation: {self.apply_nonlin}')

        # Skip background if do_bg is False
        if not self.do_bg:
            x = x[:, 1:]
            y = y[:, 1:]

        # Compute dice per channel
        if self.batch_dice:
            # Flatten spatial dimensions
            x_flat = x.view(x.shape[0], x.shape[1], -1)  # [B, C, N]
            y_flat = y.view(y.shape[0], y.shape[1], -1)  # [B, C, N]

            # Compute intersection and union per channel
            intersection = (x_flat * y_flat).sum(dim=2)  # [B, C]
            union = x_flat.sum(dim=2) + y_flat.sum(dim=2)  # [B, C]

            # Dice per channel
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # [B, C]
            dice_loss = 1.0 - dice.mean()

        else:
            # Per-sample dice
            x_flat = x.view(x.shape[0], x.shape[1], -1)
            y_flat = y.view(y.shape[0], y.shape[1], -1)

            intersection = (x_flat * y_flat).sum(dim=2)
            union = x_flat.sum(dim=2) + y_flat.sum(dim=2)

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1.0 - dice.mean()

        return dice_loss


class CrossEntropyLoss(nn.Module):
    """Cross-entropy loss wrapper for segmentation.

    Supports class weights and proper handling of segmentation targets.

    Args:
        weight: Class weights tensor or None
        ignore_index: Index to ignore in loss computation
        reduction: Reduction mode ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        weight: Union[torch.Tensor, None] = None,
        ignore_index: int = -100,
        reduction: str = 'mean',
    ) -> None:
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute CrossEntropy loss.

        Args:
            x: Network output [B, C, D, H, W] or [B, C, H, W] (logits)
            y: Ground truth [B, C, D, H, W] (one-hot) or [B, D, H, W] (class indices)

        Returns:
            CrossEntropy loss value
        """
        # Convert one-hot to class indices if needed
        if len(y.shape) == len(x.shape):
            y = y.argmax(dim=1)  # [B, C, ...] -> [B, ...]

        return self.ce_loss(x, y.long())


class TopKLoss(nn.Module):
    """TopK loss that focuses on hardest pixels.

    Computes CrossEntropy loss but only on the top-k% hardest pixels.
    Useful for imbalanced datasets.

    Args:
        k_percent: Percentage of hardest pixels to use (0.0 to 1.0)
        weight: Class weights tensor or None
        ignore_index: Index to ignore in loss computation
    """

    def __init__(
        self,
        k_percent: float = 0.2,
        weight: Union[torch.Tensor, None] = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.k_percent = k_percent
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute TopK CrossEntropy loss.

        Args:
            x: Network output [B, C, D, H, W] or [B, C, H, W] (logits)
            y: Ground truth [B, C, D, H, W] (one-hot) or [B, D, H, W] (class indices)

        Returns:
            TopK CrossEntropy loss value
        """
        # Convert one-hot to class indices if needed
        if len(y.shape) == len(x.shape):
            y_class = y.argmax(dim=1).long()
        else:
            y_class = y.long()

        # Compute per-pixel CrossEntropy loss
        ce_loss_per_pixel = F.cross_entropy(
            x, y_class, weight=self.weight, ignore_index=self.ignore_index, reduction='none'
        )

        # Flatten spatial dimensions
        ce_loss_flat = ce_loss_per_pixel.view(ce_loss_per_pixel.shape[0], -1)  # [B, N]

        # Get top-k% hardest pixels
        num_pixels = ce_loss_flat.shape[1]
        k_pixels = max(1, int(self.k_percent * num_pixels))

        # Get top-k values per batch
        topk_loss, _ = torch.topk(ce_loss_flat, k_pixels, dim=1)  # [B, k]

        # Average over top-k pixels and batch
        return topk_loss.mean()


class DiceCELoss(nn.Module):
    """Combined Dice and CrossEntropy loss.

    Weighted combination of Dice loss and CrossEntropy loss.
    Commonly used in nnUNet v2.

    Args:
        dice_weight: Weight for Dice loss component
        ce_weight: Weight for CrossEntropy loss component
        apply_nonlin: Activation for Dice ('sigmoid' or 'softmax')
        batch_dice: If True, compute dice over batch dimension
        smooth: Smoothing factor for Dice
        do_bg: If False, ignore background in Dice
        ce_weight_tensor: Class weights for CrossEntropy
        ignore_index: Index to ignore in CrossEntropy
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        apply_nonlin: Union[str, None] = 'sigmoid',
        batch_dice: bool = True,
        smooth: float = 1e-5,
        do_bg: bool = True,
        ce_weight_tensor: Union[torch.Tensor, None] = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(apply_nonlin=apply_nonlin, batch_dice=batch_dice, smooth=smooth, do_bg=do_bg)
        self.ce_loss = CrossEntropyLoss(weight=ce_weight_tensor, ignore_index=ignore_index)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute combined Dice + CrossEntropy loss.

        Args:
            x: Network output [B, C, D, H, W] or [B, C, H, W]
            y: Ground truth [B, C, D, H, W] (one-hot) or [B, D, H, W] (class indices)

        Returns:
            Combined loss value
        """
        dice = self.dice_loss(x, y)
        ce = self.ce_loss(x, y)
        return self.dice_weight * dice + self.ce_weight * ce


class DiceCELossTopK(nn.Module):
    """Combined Dice, CrossEntropy, and TopK loss.

    Weighted combination of all three loss components.

    Args:
        dice_weight: Weight for Dice loss component
        ce_weight: Weight for CrossEntropy loss component
        topk_weight: Weight for TopK loss component
        k_percent: Percentage of hardest pixels for TopK
        apply_nonlin: Activation for Dice ('sigmoid' or 'softmax')
        batch_dice: If True, compute dice over batch dimension
        smooth: Smoothing factor for Dice
        do_bg: If False, ignore background in Dice
        ce_weight_tensor: Class weights for CrossEntropy
        ignore_index: Index to ignore in CrossEntropy
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        topk_weight: float = 1.0,
        k_percent: float = 0.2,
        apply_nonlin: Union[str, None] = 'sigmoid',
        batch_dice: bool = True,
        smooth: float = 1e-5,
        do_bg: bool = True,
        ce_weight_tensor: Union[torch.Tensor, None] = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.topk_weight = topk_weight
        self.dice_loss = DiceLoss(apply_nonlin=apply_nonlin, batch_dice=batch_dice, smooth=smooth, do_bg=do_bg)
        self.ce_loss = CrossEntropyLoss(weight=ce_weight_tensor, ignore_index=ignore_index)
        self.topk_loss = TopKLoss(k_percent=k_percent, weight=ce_weight_tensor, ignore_index=ignore_index)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute combined Dice + CrossEntropy + TopK loss.

        Args:
            x: Network output [B, C, D, H, W] or [B, C, H, W]
            y: Ground truth [B, C, D, H, W] (one-hot) or [B, D, H, W] (class indices)

        Returns:
            Combined loss value
        """
        dice = self.dice_loss(x, y)
        ce = self.ce_loss(x, y)
        topk = self.topk_loss(x, y)
        return self.dice_weight * dice + self.ce_weight * ce + self.topk_weight * topk
