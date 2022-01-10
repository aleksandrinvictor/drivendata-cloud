"""Model exponential moving average."""

import math
from copy import deepcopy
from typing import Any, Tuple

import torch
import torch.nn as nn


def copy_attr(a: nn.Module, b: nn.Module, include: Tuple = (), exclude: Tuple = ()) -> None:
    """Copy attributes from b to a.

    Parameters
    ----------
    model : nn.Module
        Model with new weights.
    include : Tuple[str], optional (default=())
        Attribute names to update.
    exclude : Tuple[str], optional (default=("process_group", "reducer"))
        Attribute names to exclude.
    """
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith("_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


class ModelEMA:
    """Model Exponential Moving Average from [1]_.

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like [2]_.
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.

    Parameters
    ----------
    model : nn.Module
        Model for which to mantain moving averages.
    decay : float, optional (default=0.9999)
        Weights decay.
    updates : int, optional (default=0)
        Number of updates.

    References
    ----------
    .. [1] https://github.com/rwightman/pytorch-image-models
    .. [2] https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model: nn.Module, decay: float = 0.99998, updates: int = 0):
        """Init EMA."""
        self.ema = deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def __call__(self, x):
        return self.ema(x)

    def update(self, model: nn.Module) -> None:
        """Update EMA parameters.

        Parameters
        ----------
        model : nn.Module
            Model with new weights.
        """
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()

    def update_attr(
        self,
        model: nn.Module,
        include: Tuple = (),
        exclude: Tuple = ("process_group", "reducer"),
    ) -> None:
        """Update EMA attributes.

        Parameters
        ----------
        model : nn.Module
            Model with new weights.
        include : Tuple[str], optional (default=())
            Attribute names to update.
        exclude : Tuple[str], optional (default=("process_group", "reducer"))
            Attribute names to exclude.
        """
        copy_attr(self.ema, model, include, exclude)
