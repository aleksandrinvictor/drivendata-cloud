from typing import Any, Callable, Dict, List

import cv2
import numpy as np
from skimage import filters
from torch import Tensor

from cloud.utils import build_object


class GaussianFilter:
    def __init__(self, sigma: int = 1, mode: str = "nearest") -> None:
        self.sigma = sigma
        self.mode = mode

    def __call__(self, mask: np.ndarray) -> np.ndarray:

        prediction = np.copy(mask)

        for i in range(prediction.shape[0]):
            prediction[i] = filters.gaussian(mask[i], sigma=self.sigma, mode=self.mode)

        return prediction


class MedianFilter:
    def __init__(self, mode: str = "nearest") -> None:
        self.mode = mode

    def __call__(self, mask: np.ndarray) -> np.ndarray:

        prediction = np.copy(mask)

        for i in range(prediction.shape[0]):
            prediction[i] = filters.median(mask[i], mode=self.mode)

        return prediction


# class ConvexHull:
#     def __init__(self, prob_low_bound: float = 0.2, prob_high_bound: float = 0.8):
#         self.prob_low_bound = prob_low_bound
#         self.prob_high_bound = prob_high_bound

#     def __len__(self):
#         return 1

#     def __call__(self, mask: np.ndarray) -> np.ndarray:

#         prediction = np.copy(mask)
#         kernel = np.ones((2, 2), np.uint8)

#         for i in range(prediction.shape[0]):

#             edges = filters.sobel(mask[i])

#             unreliable_pixels = np.zeros(edges.shape, dtype=np.uint8)
#             unreliable_pixels[(edges > 0.2) & (edges < 0.8)] = 255

#             unreliable_pixels = cv2.erode(unreliable_pixels, kernel, iterations=1)

#             contours, _ = cv2.findContours(np.copy(unreliable_pixels), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#             for contour in contours:
#                 contour = cv2.convexHull(contour)

#                 cv2.drawContours(prediction[i], [contour], 0, (0, 0, 0), 3)

#         return prediction


class MorphologicalTransform:
    def __init__(self, kernel_size: int) -> None:
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)


class Opening(MorphologicalTransform):
    def __init__(self, kernel_size: int) -> None:
        super().__init__(kernel_size)

    def __call__(self, input: np.ndarray) -> np.ndarray:
        for i in range(input.shape[0]):
            input[i] = cv2.morphologyEx(input[i], cv2.MORPH_OPEN, self.kernel)

        return input


class Closing(MorphologicalTransform):
    def __init__(self, kernel_size: int) -> None:
        super().__init__(kernel_size)

    def __call__(self, input: np.ndarray) -> np.ndarray:
        for i in range(input.shape[0]):
            input[i] = cv2.morphologyEx(input[i], cv2.MORPH_CLOSE, self.kernel)

        return input


class Erode(MorphologicalTransform):
    def __init__(self, kernel_size: int, iterations: int = 1) -> None:
        super().__init__(kernel_size)

        self.iterations = iterations

    def __call__(self, input: np.ndarray) -> np.ndarray:
        for i in range(input.shape[0]):
            input[i] = cv2.erode(input[i], self.kernel, iterations=self.iterations)

        return input


class Dilate(MorphologicalTransform):
    def __init__(self, kernel_size: int, iterations: int = 1) -> None:
        super().__init__(kernel_size)

        self.iterations = iterations

    def __call__(self, input: np.ndarray) -> np.ndarray:
        for i in range(input.shape[0]):
            input[i] = cv2.dilate(input[i], self.kernel, iterations=self.iterations)

        return input


class TripletRule:
    def __init__(
        self,
        image_size: int,
        kernel_size: int,
        min_area: int = 2,
        patches_threshold: float = 0.8,
        pixel_threshold: float = 0.4,
    ) -> None:

        self.image_size = image_size

        self.crop = 0
        if image_size % kernel_size != 0:
            self.crop = image_size % kernel_size // 2

        self.crop_size = self.image_size - self.crop * 2

        self.kernel_size = kernel_size
        self.area = kernel_size * kernel_size

        self.min_area = min_area
        self.patches_threshold = patches_threshold
        self.pixel_threshold = pixel_threshold

    def _process_mask(self, mask: np.ndarray) -> np.ndarray:
        cropped_mask = mask[self.crop : self.image_size - self.crop, self.crop : self.image_size - self.crop]

        patches = self._split(cropped_mask)

        conf_pixels_sum = (patches > self.patches_threshold).sum(axis=(1, 2))
        conf_patches_idx = conf_pixels_sum >= self.min_area
        confident_patches = patches[conf_patches_idx]

        confident_patches[confident_patches > self.pixel_threshold] = 1.0
        confident_patches[confident_patches <= self.pixel_threshold] = 0.0
        patches[conf_patches_idx] = confident_patches
        patches[~conf_patches_idx] = 0.0

        mask[self.crop : self.image_size - self.crop, self.crop : self.image_size - self.crop] = self._assebmle(patches)

        return mask

    def _split(self, x: np.ndarray) -> np.ndarray:
        """Split a matrix into sub-matrices."""

        r, h = x.shape
        return (
            x.reshape(h // self.kernel_size, self.kernel_size, -1, self.kernel_size)
            .swapaxes(1, 2)
            .reshape(-1, self.kernel_size, self.kernel_size)
        )

    def _assebmle(self, patches: np.ndarray) -> np.ndarray:
        patches = patches.reshape(
            self.crop_size // self.kernel_size, self.crop_size // self.kernel_size, self.kernel_size, self.kernel_size
        )

        return np.transpose(patches, (0, 2, 1, 3)).reshape(self.crop_size, -1)

    def __call__(self, input: np.ndarray) -> np.ndarray:

        for i in range(input.shape[0]):
            input[i] = self._process_mask(input[i])

        return input


class PostProcess:
    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.transforms: List[Callable] = []

        for transform in cfg:
            self.transforms.append(build_object(transform))

    def __call__(self, input: Tensor) -> Tensor:

        temp = input.copy()

        for transform in self.transforms:
            temp = transform(temp)

        return temp
