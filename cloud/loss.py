import torch
from torch import Tensor

from .metrics import activation_mapping, dice_score


class SoftDiceLoss:
    def __init__(
        self,
        activation: str = None,
        eps: float = 1e-6,
    ) -> None:
        self.eps = eps

        self.activation = None

        if activation is not None:
            if activation in activation_mapping.keys():
                self.activation = activation_mapping[activation]
            else:
                raise ValueError("Unknown activation, " f"should be one of {activation_mapping.keys()}")

    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
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
            soft dice loss
        """
        if self.activation is not None:
            input = self.activation(input)

        return 1 - dice_score(input, target, self.eps)


class BceDiceLoss:
    def __init__(self, bce_coef: float = 0.5, dice_coef: float = 0.5):

        self.bce = torch.nn.CrossEntropyLoss()
        self.dice = SoftDiceLoss(activation="softmax")

        self.bce_coef = bce_coef
        self.dice_coef = dice_coef

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Parameters
        ----------
        y_pred: Tensor
            input tensor with shape (batch_size, num_classes, height, width)
            must sum to 1 over c channel (such as after softmax)
        y_true: Tensor
            one hot target tensor with shape
            (batch_size, num_classes, height, width)

        Returns
        -------
        Tensor
            bce dice loss
        """
        return self.bce_coef * self.bce(y_pred, torch.squeeze(y_true, dim=1)) + self.dice_coef * self.dice(
            y_pred, y_true
        )


class CrossEntropyLoss:
    def __init__(self):

        self.ce = torch.nn.CrossEntropyLoss()

    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Parameters
        ----------
        y_pred: Tensor
            input tensor with shape (batch_size, num_classes, height, width)
            must sum to 1 over c channel (such as after softmax)
        y_true: Tensor
            one hot target tensor with shape
            (batch_size, num_classes, height, width)

        Returns
        -------
        Tensor
            bce dice loss
        """
        return self.ce(y_pred, torch.squeeze(y_true, dim=1))
