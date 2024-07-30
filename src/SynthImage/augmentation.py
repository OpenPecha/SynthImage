import math
from enum import Enum

import cv2
import numpy as np
from PIL import Image, ImageOps


class DistortionMode(Enum):
    """A simple selection for the mode used for font contour distortion"""

    additive = 0
    subtractive = 1


def distort_line(
    image: np.ndarray,
    mode,
    edge_tresh1,
    edge_tresh2,
    kernel_width,
    kernel_height,
    kernel_iterations,
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


class WaveDeformer:
    def __init__(self, grid, multiplier, offset) -> None:
        self.multiplier = multiplier
        self.offset = offset
        self.grid = grid

    def transform(self, x, y):
        """Applies a wave transformation to the y-coordinate based on the x-coordinate.

        Args:
            x (float): The x-coordinate.
            y (float): The y-coordinate.

        Returns:
            tuple: A tuple (x, y) with the transformed coordinates.
        """
        y = y + self.multiplier * math.sin(x / self.offset)
        return x, y

    def transform_rectangle(self, x0, y0, x1, y1):
        """Transforms the coordinates of a rectangle by applying the wave transformation.

        Args:
            x0 (float): The x-coordinate of the top-left corner.
            y0 (float): The y-coordinate of the top-left corner.
            x1 (float): The x-coordinate of the bottom-right corner.
            y1 (float): The y-coordinate of the bottom-right corner.

        Returns:
            tuple: A tuple containing the transformed coordinates of the rectangle.
        """
        return (
            *self.transform(x0, y0),
            *self.transform(x0, y1),
            *self.transform(x1, y1),
            *self.transform(x1, y0),
        )

    def getmesh(self, img):
        """Generates a mesh for the image by applying the wave transformation.

        Args:
            img (PIL.Image.Image): The image for which to generate the mesh.

        Returns:
            list:A list of tuples where each tuple contains the target and source coordinates for the mesh.
        """
        self.w, self.h = img.size
        gridspace = self.grid
        target_grid = []
        for x in range(0, self.w, gridspace):
            for y in range(0, self.h, gridspace):
                target_grid.append((x, y, x + gridspace, y + gridspace))
        source_grid = [self.transform_rectangle(*rect) for rect in target_grid]
        return [t for t in zip(target_grid, source_grid)]


def deform_image(image: np.ndarray, grid_size, multiplier, offset):
    """Applies a wave-like deformation to an input image.


    Args:
        image (np.ndarray):The input image as a NumPy array.
        grid_size (_type_): The size of the grid for the deformation.
        multiplier (_type_): The multiplier for the wave amplitude.
        offset (_type_): The offset for the wave frequency.

    Returns:
        np.ndarray:  The deformed image as a NumPy array.
    """
    deformer = WaveDeformer(grid=grid_size, multiplier=multiplier, offset=offset)
    pil_image = Image.fromarray(image)
    deformed_img = ImageOps.deform(pil_image, deformer)
    return np.array(deformed_img)


class Augmentation:
    def __init__(
        self,
        original_img_obj,
        is_deformed: bool = True,
        is_distort: bool = True,
        has_background: bool = False,
        is_torn: bool = True,
        is_dirty: bool = True,
    ):
        self.original_img_obj = original_img_obj
        self.is_deformed = is_deformed
        self.is_distort = is_distort
        self.has_background = has_background
        self.is_torn = is_torn
        self.is_dirty = is_dirty

    def apply_augmentation(self):
        """Apply Augmentation to the original Image"""
        aug_img = self.original_img_obj
        if self.is_deformed:
            aug_img = self.apply_deform(aug_img)
        if self.is_distort:
            aug_img = self.apply_distort(aug_img)

    def apply_deform(
        self, aug_img, grid: int = 20, multiplier: int = 6, offset: int = 70
    ):
        """Applies a wave-like deformation to the given image.

        Args:
            aug_img (PIL.Image.Image): The input image to be deformed.
            grid (int, optional): The size of the grid for the deformation. Defaults to 20.
            multiplier (int, optional): The multiplier for the wave amplitude. Defaults to 6.
            offset (int, optional): The offset for the wave frequency. Defaults to 70.

        Returns:
            PIL.Image.Image: The deformed image as a PIL Image.
        """
        page_img_np = np.array(aug_img)
        deformed_page_img = deform_image(page_img_np, grid, multiplier, offset)
        return Image.fromarray(deformed_page_img)

    def apply_distort(
        self,
        aug_img,
        mode: int = 1,
        edge_tresh1: int = 100,
        edge_tresh2: int = 200,
        kernel_width: int = 2,
        kernel_height: int = 1,
        kernel_iterations=2,
    ):
        """Applies a distortion effect to the given image using edge detection and morphological transformations.

        Args:
            aug_img (PIL.Image.Image): The input image to be distorted.
            mode (int, optional): The distortion mode, either additive or subtractive. Defaults to DistortionMode.additive.
            edge_tresh1 (int, optional): The first threshold for the hysteresis procedure in the Canny edge detection. Defaults to 100.
            edge_tresh2 (int, optional): The second threshold for the hysteresis procedure in the Canny edge detection. Defaults to 200.
            kernel_width (int, optional): The width of the kernel used for morphological transformations. Defaults to 2.
            kernel_height (int, optional): The height of the kernel used for morphological transformations. Defaults to 1.
            kernel_iterations (int, optional): The number of iterations for the morphological transformations. Defaults to 2.

        Returns:
            PIL.Image.Image: The distorted image as a PIL Image.
        """  # noqa
        page_img_np = np.array(aug_img)
        distorted_img = distort_line(
            page_img_np,
            mode,
            edge_tresh1,
            edge_tresh2,
            kernel_width,
            kernel_height,
            kernel_iterations,
        )
        return Image.fromarray(distorted_img)
