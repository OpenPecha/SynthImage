import math

import numpy as np
from PIL import Image, ImageOps


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


class DeformAugmentation:
    def __init__(
        self, original_img_obj, grid: int = 20, multiplier: int = 6, offset: int = 70
    ):
        """Initializes the DeformAugmentation class with the given parameters.

        Args:
            original_img_obj (_type_): The input image to be deformed.
            grid (int, optional): The size of the grid for the deformation. Defaults to 20.
            multiplier (int, optional): The multiplier for the wave amplitude. Defaults to 6.
            offset (int, optional): The offset for the wave frequency. Defaults to 70.
        """
        self.original_img_obj = original_img_obj
        self.grid = grid
        self.multiplier = multiplier
        self.offset = offset

    def apply(self):
        """Applies a wave-like deformation to the given image.

        Returns:
            PIL.Image.Image: The deformed image as a PIL Image.
        """
        aug_img = self.original_img_obj
        page_img_np = np.array(aug_img)
        deformer = WaveDeformer(
            grid=self.grid, multiplier=self.multiplier, offset=self.offset
        )
        pil_image = Image.fromarray(page_img_np)
        deformed_img = ImageOps.deform(pil_image, deformer)
        deformed_page_img = np.array(deformed_img)
        return Image.fromarray(deformed_page_img)
