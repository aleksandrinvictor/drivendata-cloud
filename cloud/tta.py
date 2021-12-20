from typing import Any, Callable, Dict, List

import torchvision.transforms.functional as TF
from torch import Tensor

from cloud.utils import build_object


class Flip:
    def __init__(self, orientation: str = "horizontal"):

        if orientation == "horizontal":
            self.transform = TF.hflip
        else:
            self.transform = TF.vflip

    def __len__(self):
        return 1

    def __call__(self, model: Callable, input: Tensor, add_model_pred: bool = False) -> Tensor:

        divide_by = len(self)

        pred = self.transform(model(self.transform(input)))

        if add_model_pred:
            pred += model(input)
            divide_by += 1

        return pred / divide_by


class Rotate:
    def __init__(self, angles: List[float]):

        self.angles = angles

    def __len__(self):
        return len(self.angles)

    def __call__(self, model: Callable, input: Tensor, add_model_pred: bool = False) -> Tensor:

        pred = None
        divide_by = len(self)

        if add_model_pred:
            pred = model(input)
            divide_by += 1
        else:
            pred = TF.rotate(
                model(TF.rotate(input, angle=self.angles[0])),
                angle=360 - self.angles[0],
            )

        for angle in self.angles[1:]:
            pred += TF.rotate(model(TF.rotate(input, angle=angle)), angle=360 - angle)

        return pred / divide_by


class TTA:
    def __init__(self, cfg: Dict[str, Any]):

        self.transforms: List[Callable] = []

        for aug in cfg:
            self.transforms.append(build_object(aug))

    def __len__(self):
        return len(self.transforms)

    def __call__(self, model: Callable, input: Tensor) -> Tensor:
        pred = model(input)

        for transform in self.transforms:
            temp = transform(model, input)
            pred += temp

        return pred / (len(self) + 1)
