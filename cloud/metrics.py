import torch
import torch.nn as nn
from torch import Tensor

activation_mapping = {"softmax": nn.Softmax(dim=1), "sigmoid": nn.Sigmoid}


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold) * 1
    else:
        return x


class IoU:
    def __init__(
        self,
        eps: float = 1e-7,
        activation: str = None,
        threshold: float = 0.5,
    ) -> None:

        self.threshold = threshold
        self.eps = eps

        if activation:
            self.activation = activation_mapping[activation]
        else:
            self.activation = None

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:

        if self.activation is not None:
            input = self.activation(input)
            input = input[:, 1].unsqueeze(dim=1)

        input = _threshold(input, threshold=self.threshold)

        return jaccard_score(input, target, eps=self.eps)


class Dice:
    def __init__(
        self,
        eps: float = 1e-7,
        activation: str = None,
        threshold: float = 0.5,
    ) -> None:

        self.threshold = threshold
        self.eps = eps

        self.activation = activation

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:

        if self.activation is not None:
            activation = activation_mapping[self.activation]

            input = activation(input)

        input = _threshold(input, threshold=self.threshold)

        return dice_score(input, target, eps=self.eps)


def jaccard_score(input: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Parameters
    ----------
    input: Tensor
        input tensor with shape (batch_size, num_classes, height, width)
        must sum to 1 over c channel (such as after softmax)
    target: Tensor
        one hot target tensor with shape
        (batch_size, num_classes, height, width)

    Returns
    -------
    Tensor
        mean jaccard score
    """
    intersection = torch.sum(input * target, dim=(2, 3))
    union = torch.sum(input, dim=(2, 3)) + torch.sum(target, dim=(2, 3)) - intersection
    return torch.mean((intersection + eps) / (union + eps))


def dice_score(input: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """
    Parameters
    ----------
    input: Tensor
        input tensor with shape (batch_size, num_classes, height, width)
        must sum to 1 over c channel (such as after softmax)
    target: Tensor
        one hot target tensor with shape
        (batch_size, num_classes, height, width)

    Returns
    -------
    Tensor
        mean dice score
    """
    numerator = 2 * torch.sum(input * target, dim=(2, 3))
    denominator = torch.sum(input, dim=(2, 3)) + torch.sum(target, dim=(2, 3))

    return torch.mean((numerator + eps) / (denominator + eps))
