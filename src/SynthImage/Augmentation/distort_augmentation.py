from enum import Enum

import cv2
import numpy as np
from PIL import Image


class DistortionMode(Enum):
    """A simple selection for the mode used for font contour distortion"""

    additive = 0
    subtractive = 1


class DistortAugmentation:
    def __init__(
        self,
        original_img_obj,
        mode: DistortionMode = DistortionMode.additive,
        edge_tresh1: int = 100,
        edge_tresh2: int = 200,
        kernel_width: int = 2,
        kernel_height: int = 1,
        kernel_iterations=2,
    ):
        """Initializes the DistortAugmentation class with the given parameters.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be distorted.
            mode (DistortionMode, optional): The distortion mode, either additive or subtractive. Defaults to DistortionMode.additive.
            edge_tresh1 (int, optional): The first threshold for the hysteresis procedure in the Canny edge detection. Defaults to 100.
            edge_tresh2 (int, optional): The second threshold for the hysteresis procedure in the Canny edge detection. Defaults to 200.
            kernel_width (int, optional): The width of the kernel used for morphological transformations. Defaults to 2.
            kernel_height (int, optional): The height of the kernel used for morphological transformations. Defaults to 1.
            kernel_iterations (int, optional): The number of iterations for the morphological transformations. Defaults to 2.
        """  # noqa
        self.original_img_obj = original_img_obj
        self.mode = mode
        self.edge_tresh1 = edge_tresh1
        self.edge_tresh2 = edge_tresh2
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height
        self.kernel_iterations = kernel_iterations

    def apply(self):
        """Applies a distortion effect to the given image using edge detection and morphological transformations.

        Returns:
            PIL.Image.Image: The distorted image as a PIL Image.
        """
        if type(self.original_img_obj) is not np.array:
            page_img_np = np.array(self.original_img_obj)
        edges = cv2.Canny(page_img_np, self.edge_tresh1, self.edge_tresh2)
        if edges is None:
            return page_img_np
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        kernel = np.ones((self.kernel_width, self.kernel_height), np.uint8)
        edges = cv2.erode(edges, kernel, iterations=self.kernel_iterations)
        edges = cv2.dilate(edges, kernel, iterations=self.kernel_iterations)
        indices = np.where(edges[:, :] == 255)
        cv_image_added = page_img_np.copy()
        if self.mode == DistortionMode.additive:
            cv_image_added[indices[0], indices[1], :] = [0]
        else:
            cv_image_added[indices[0], indices[1], :] = [255]
        return Image.fromarray(cv_image_added)
