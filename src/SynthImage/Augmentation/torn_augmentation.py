import random

from PIL import Image, ImageDraw


class TornAugmentation:
    def __init__(
        self,
        original_img_obj,
        num_tears: int = 5,
        tear_size: int = 30,
        jagged_step: int = 5,
        jagged_variability: int = 5,
    ):
        """Initializes the TornAugmentation class with the provided image and parameters.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
            num_tears (int, optional): The number of tears to apply to the image. Defaults to 5.
            tear_size (int, optional): The size of each tear in pixels. Defaults to 30.
            jagged_step (int, optional): The step size for jagged line creation. Defaults to 5.
            jagged_variability (int, optional): The variability in jagged line positioning. Defaults to 5.
        """
        self.original_img_obj = original_img_obj
        self.num_tears = num_tears
        self.tear_size = tear_size
        self.jagged_step = jagged_step
        self.jagged_variability = jagged_variability

    def apply_torn(self):
        """Applies a torn effect to the original image.

        Returns:
            PIL.Image.Image: The image with the torn effect applied.
        """
        aug_img = self.original_img_obj

        if aug_img.mode != "RGBA":
            aug_img = aug_img.convert("RGBA")

        width, height = aug_img.size
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        # Define areas for tears
        for _ in range(self.num_tears):
            edge = random.choice(["top", "bottom", "left", "right"])

            if edge == "top":
                x0 = random.randint(0, width - self.tear_size)
                y0 = 0
                jagged_line = [
                    (
                        x0
                        + i
                        + random.randint(
                            -self.jagged_variability, self.jagged_variability
                        ),
                        y0
                        + random.randint(
                            -self.jagged_variability, self.jagged_variability
                        ),
                    )
                    for i in range(0, self.tear_size, self.jagged_step)
                ]
            elif edge == "bottom":
                x0 = random.randint(0, width - self.tear_size)
                y0 = height - self.tear_size
                jagged_line = [
                    (
                        x0
                        + i
                        + random.randint(
                            -self.jagged_variability, self.jagged_variability
                        ),
                        y0
                        + random.randint(
                            -self.jagged_variability, self.jagged_variability
                        ),
                    )
                    for i in range(0, self.tear_size, self.jagged_step)
                ]
            elif edge == "left":
                x0 = 0
                y0 = random.randint(0, height - self.tear_size)
                jagged_line = [
                    (
                        x0
                        + random.randint(
                            -self.jagged_variability, self.jagged_variability
                        ),
                        y0
                        + i
                        + random.randint(
                            -self.jagged_variability, self.jagged_variability
                        ),
                    )
                    for i in range(0, self.tear_size, self.jagged_step)
                ]
            elif edge == "right":
                x0 = width - self.tear_size
                y0 = random.randint(0, height - self.tear_size)
                jagged_line = [
                    (
                        x0
                        + random.randint(
                            -self.jagged_variability, self.jagged_variability
                        ),
                        y0
                        + i
                        + random.randint(
                            -self.jagged_variability, self.jagged_variability
                        ),
                    )
                    for i in range(0, self.tear_size, self.jagged_step)
                ]

            draw.line(jagged_line, fill=255, width=self.tear_size)

        # Apply the mask to the image
        torn_image = aug_img.copy()
        torn_image.paste((0, 0, 0, 0), mask=mask)
        return torn_image
