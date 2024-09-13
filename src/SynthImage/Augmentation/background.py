import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class BackgroundAugmentation:
    def __init__(
        self,
        original_img_obj: Image.Image,
        background_folder: str = "./data/background",
    ):
        """Initialize the BackgroundAugmentation object.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
            background_folder (str): Path to the folder containing background images.
        """
        self.original_img_obj = original_img_obj
        self.background_folder = background_folder

    def apply(self) -> Image.Image:
        """Apply background augmentation to the input image.

        Returns:
            PIL.Image.Image: The augmented image with a new background.
        """
        background_images = list(Path(self.background_folder).glob("*"))
        if not background_images:
            raise ValueError("No background images found in the specified folder.")

        # Select a random background image
        background_image_path = random.choice(background_images)
        background_image = Image.open(background_image_path).convert("RGB")

        # Resize the background to match the dimensions of the synthetic page
        background_image = background_image.resize(
            self.original_img_obj.size, Image.LANCZOS
        )

        # Convert images to numpy arrays
        image_np = np.array(self.original_img_obj)
        background_np = np.array(background_image)

        # Create a mask for the text (assuming white text on a dark background)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        _, text_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Invert the mask to get the text region
        text_mask_inv = cv2.bitwise_not(text_mask)

        # Extract the text from the synthetic page
        text_extracted = cv2.bitwise_and(image_np, image_np, mask=text_mask_inv)

        # Extract background where the text will not be placed
        background_torn = cv2.bitwise_and(background_np, background_np, mask=text_mask)

        # Blend text and background (alpha blending)
        blended_image = self.blend_images(
            text_extracted, background_torn, text_mask_inv
        )

        return Image.fromarray(blended_image)

    def blend_images(self, text_image, background_image, mask):
        """Blend the text image and background image using the mask.

        Args:
            text_image (np.ndarray): The image containing the extracted text.
            background_image (np.ndarray): The background image.
            mask (np.ndarray): The mask indicating where the text should be.

        Returns:
            np.ndarray: The blended image.
        """
        # Normalize the mask to have values between 0 and 1
        alpha_mask = mask.astype(float) / 255.0

        # Blend the text and the background based on the mask
        blended = (
            alpha_mask[..., None] * text_image
            + (1 - alpha_mask[..., None]) * background_image
        ).astype(np.uint8)

        return blended
