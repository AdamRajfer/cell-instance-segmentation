import torch
from torch.nn import functional as F


def binary_crossentropy(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(pred, true, reduction="none").mean([1, 2])


def jaccard_loss(pred: torch.Tensor, true: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    intersection = (pred * true).sum([1, 2])
    total = (pred + true).sum([1, 2])
    union = total - intersection
    jaccard = (intersection + smooth) / (union + smooth)
    return 1 - jaccard


def dice_loss(pred: torch.Tensor, true: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    intersection = (pred * true).sum([1, 2])
    total = (pred + true).sum([1, 2])
    dice = (2 * intersection + smooth) / (total + smooth)
    return 1 - dice


def focal_loss(pred: torch.Tensor, true: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    bce = binary_crossentropy(pred, true)
    return alpha * (1 - torch.exp(-bce)) ** gamma * bce


def tversky_loss(
    pred: torch.Tensor, true: torch.Tensor, smooth: float = 1.0, alpha: float = 0.3, beta: float = 0.7
) -> torch.Tensor:
    intersection = (pred * true).sum([1, 2])
    fp = ((1 - true) * pred).sum([1, 2])
    fn = (true * (1 - pred)).sum([1, 2])
    tversky = (intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth)
    return 1 - tversky


def jaccard_binary_crossentropy_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return jaccard_loss(pred, true) + binary_crossentropy(pred, true)
