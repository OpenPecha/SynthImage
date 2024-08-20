import random

import numpy as np
from PIL import Image, ImageDraw


class DirtySpotAugmentation:
    def __init__(self, original_img_obj, dirty_spots):
        """Initialize the DirtySpotAugmentation object.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
            dirty_spots (list of tuples, optional): List of tuples specifying dirty spots.
                                                    Each tuple should contain (x, y, size).
                                                    Defaults to None.
        """
        self.original_img_obj = original_img_obj
        self.dirty_spots = dirty_spots

    def apply_dirty(self):
        """Apply dirty spot augmentation to the input image.

        Returns:
            PIL.Image.Image: The augmented image with dirty spots.
        """
        img_np = np.array(self.original_img_obj)
        height, width, _ = img_np.shape

        for spot in self.dirty_spots:
            x, y, size = spot
            # Ensure the spot's position and size are within the image bounds
            x = min(max(x, 0), width - 1)
            y = min(max(y, 0), height - 1)
            size = min(max(size, 1), min(height, width))

            ellipse_width = random.randint(size // 2, size)
            ellipse_height = random.randint(size // 2, size)

            # Create an image with the dirty spot
            dirty_spot = Image.new("L", (ellipse_width, ellipse_height), 0)
            draw = ImageDraw.Draw(dirty_spot)
            draw.ellipse((0, 0, ellipse_width, ellipse_height), fill=255)

            # Convert to numpy array
            dirty_spot_np = np.array(dirty_spot)

            # Define region of interest in the original image
            x1 = max(0, x - ellipse_width // 2)
            y1 = max(0, y - ellipse_height // 2)
            x2 = min(width, x + ellipse_width // 2)
            y2 = min(height, y + ellipse_height // 2)

            # Define the region of the dirty spot that fits within the image
            spot_x1 = max(0, ellipse_width // 2 - x)
            spot_y1 = max(0, ellipse_height // 2 - y)
            spot_x2 = spot_x1 + (x2 - x1)
            spot_y2 = spot_y1 + (y2 - y1)

            # Create a mask for the dirty spot
            mask = np.zeros((height, width), dtype=np.uint8)
            if (y2 - y1) > 0 and (x2 - x1) > 0:
                dirty_spot_region = dirty_spot_np[spot_y1:spot_y2, spot_x1:spot_x2]
                mask[y1:y2, x1:x2] = dirty_spot_region

            # Apply the dirty spot to the image
            img_np[:, :, 0] = np.where(
                mask > 0, 0, img_np[:, :, 0]
            )  # Apply dirty spot to Red channel
            img_np[:, :, 1] = np.where(
                mask > 0, 0, img_np[:, :, 1]
            )  # Apply dirty spot to Green channel
            img_np[:, :, 2] = np.where(
                mask > 0, 0, img_np[:, :, 2]
            )  # Apply dirty spot to Blue channel

        return Image.fromarray(img_np)
