from typing import Tuple

import numpy as np
import torch
from torch import Tensor


def rand_bbox(image_shape: torch.Size, lam: float) -> Tuple[int, int, int, int]:
    """Return random box.
    Cut bbox from image sampling crops's center uniformly.
    Parameters
    ----------
    image_shape: torch.Size
        Shape of the batch of images [N, C, H, W].
    lam: float
        Defines ratio of the image to be cropped.
        ratio = sqrt(1 - lam).
    """
    W = image_shape[2]
    H = image_shape[3]
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)

    # Center of the crop
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class CutMix:
    """Cutmix augmentation.
    Parameters
    ----------
    beta: float, optional (default=0.2)
        Beta distribution parameter.
    lam_min: float, optional (default=0.3)
        Beta distribution sample min.
    lam_max: float, optional (default=0.4)
        Beta distribution sample max.
    task: str, optional (default="semantic_segmentation")
        Target task.
    p: float, optional (default=0.5)
        Probability of applying transform.
    """

    def __init__(
        self,
        beta: float = 0.2,
        lam_min: float = 0.01,
        lam_max: float = 0.02,
        num_boxes: int = 4,
        p: float = 0.5,
    ):
        """Init cutmix augmentation."""
        self.beta = beta
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.num_boxes = num_boxes
        self.p = p

    def __call__(self, images: Tensor, targets: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply cutmix augmentation for semantic segmentation.
        Parameters
        ----------
        images: Tensor
            Batch of images [N, C, H, W].
        targets: Tensor
            Batch of masks [N, H, W].
        """
        num_images = images.size(0)

        targets = targets.long()

        if np.random.uniform() >= self.p or num_images == 1:
            return images, targets

        rand_index = torch.randperm(images.size(0))
        mixed_images = images.clone()
        mixed_targets = targets.clone()

        for _ in range(self.num_boxes):
            lam = np.clip(np.random.beta(self.beta, self.beta), self.lam_min, self.lam_max)

            x1, y1, x2, y2 = rand_bbox(images.shape, lam)

            mixed_images[:, :, y1:y2, x1:x2] = images[rand_index, :, y1:y2, x1:x2]

            mixed_targets[:, :, y1:y2, x1:x2] = targets[rand_index, :, y1:y2, x1:x2]

        return mixed_images, mixed_targets
