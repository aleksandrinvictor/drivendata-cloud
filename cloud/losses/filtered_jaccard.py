import torch
from torch import nn


def filtered_jaccard_loss(y_pred, y_true, epsilon=1e-7):
    """
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the channels_last format.

    Parameters
    ----------
    y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
    y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
    epsilon: Used for numerical stability to avoid divide by zero errors
    """
    if y_true.sum() == 0:
        inter = ((1.0 - y_true) * (1.0 - y_pred)).sum().float()
        union = ((1.0 - y_true) + (1.0 - y_pred)).sum().float()
    else:
        inter = (y_true * y_pred).sum().float()
        union = (y_true + y_pred).sum().float()

    loss = 1.0 - (inter / (union - inter + epsilon))
    return loss


class FilteredJaccardLoss:
    def __init__(self):
        self.softmax = nn.Softmax(dim=1)

    def __call__(self, y_pred, y_true):
        # print(f"y_pred: {y_pred.shape}")
        # print(f"y_true: {y_true.shape}")
        y_true_ohe = torch.nn.functional.one_hot(y_true, num_classes=2).squeeze().permute(0, 3, 1, 2)

        # print(f"y_true_ohe: {y_true_ohe.shape}")

        return filtered_jaccard_loss(self.softmax(y_pred), y_true_ohe)
