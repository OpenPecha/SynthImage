import math

import numpy as np
from PIL import Image, ImageOps


class WaveDeformer:
    def __init__(self, grid: int = 20, multiplier: int = 6, offset: int = 70) -> None:
        self.multiplier = multiplier
        self.offset = offset
        self.grid = grid

    def transform(self, x, y):
        y = y + self.multiplier * math.sin(x / self.offset)
        return x, y

    def transform_rectangle(self, x0, y0, x1, y1):
        return (
            *self.transform(x0, y0),
            *self.transform(x0, y1),
            *self.transform(x1, y1),
            *self.transform(x1, y0),
        )

    def getmesh(self, img):
        self.w, self.h = img.size
        gridspace = self.grid
        target_grid = []
        for x in range(0, self.w, gridspace):
            for y in range(0, self.h, gridspace):
                target_grid.append((x, y, x + gridspace, y + gridspace))
        source_grid = [self.transform_rectangle(*rect) for rect in target_grid]
        return [t for t in zip(target_grid, source_grid)]


def deform_image(image: np.ndarray, grid_size, multiplier, offset):
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
        aug_img = self.original_img_obj
        if self.is_deformed:
            aug_img = self.apply_deform()
        return aug_img

    def apply_deform(self, grid: int = 20, multiplier: int = 6, offset: int = 70):
        page_img_np = np.array(self.original_img_obj)
        deformed_page_img = deform_image(page_img_np, grid, multiplier, offset)
        return Image.fromarray(deformed_page_img)
