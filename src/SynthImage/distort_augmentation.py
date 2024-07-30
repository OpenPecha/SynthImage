from enum import Enum

import cv2
import numpy as np
from PIL import Image


class DistortionMode(Enum):
    """A simple selection for the mode used for font contour distortion"""

    additive = 0
    subtractive = 1


def distort_line(
    image: np.ndarray,
    mode: DistortionMode,
    edge_tresh1: int,
    edge_tresh2: int,
    kernel_width: int,
    kernel_height: int,
    kernel_iterations: int,
):
    """Applies a line distortion effect to the given image using edge detection and morphological transformations.

    Args:
        image (np.array): The input image to be distorted.
        mode (int): The distortion mode, determining how the edges are modified.
        edge_tresh1 (int): The first threshold for the hysteresis procedure in the Canny edge detection.
        edge_tresh2 (int): The second threshold for the hysteresis procedure in the Canny edge detection.
        kernel_width (int): The width of the kernel used for morphological transformations (erosion and dilation).
        kernel_height (int): The height of the kernel used for morphological transformations (erosion and dilation).
        kernel_iterations (int): The number of iterations for the morphological transformations.


    Returns:
        np.array: The distorted image as a NumPy array.
    """
    if type(image) is not np.array:
        image = np.array(image)
    edges = cv2.Canny(image, edge_tresh1, edge_tresh2)
    if edges is None:
        return image
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    kernel = np.ones((kernel_width, kernel_height), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=kernel_iterations)
    edges = cv2.dilate(edges, kernel, iterations=kernel_iterations)
    indices = np.where(edges[:, :] == 255)
    cv_image_added = image.copy()
    if mode == DistortionMode.additive:
        cv_image_added[indices[0], indices[1], :] = [0]
    else:
        cv_image_added[indices[0], indices[1], :] = [255]
    return cv_image_added


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

    def apply_distort(self):
        """Applies a distortion effect to the given image using edge detection and morphological transformations.

        Returns:
            PIL.Image.Image: The distorted image as a PIL Image.
        """
        page_img_np = np.array(self.original_img_obj)
        distorted_img = distort_line(
            page_img_np,
            self.mode,
            self.edge_tresh1,
            self.edge_tresh2,
            self.kernel_width,
            self.kernel_height,
            self.kernel_iterations,
        )
        return Image.fromarray(distorted_img)
