import numpy as np
from augraphy import TextureGenerator
from PIL import Image


class PaperTextureAugmentation:
    def __init__(
        self,
        original_img_obj,
        texture_type="rough_stains",
        texture_width=None,
        texture_height=None,
        quilt_texture=0,
        alpha=0.7,
        beta=0.3,
        num_channels=3,
    ):
        """Initialize the PaperTextureAugmentation class with an image and parameters.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
            texture_type (str): Type of texture to generate (e.g., "rough_stains").
            texture_width (int): Width of the texture image.
            texture_height (int): Height of the texture image.
            quilt_texture (int): Quilt texture setting.
            alpha (float): Blending coefficient for the original image.
            beta (float): Blending coefficient for the texture.
            num_channels (int): Number of channels to expand the texture to (e.g., 3 for RGB).
        """
        self.original_img_obj = original_img_obj
        self.texture_type = texture_type
        self.texture_width = texture_width
        self.texture_height = texture_height
        self.quilt_texture = quilt_texture
        self.alpha = alpha
        self.beta = beta
        self.num_channels = num_channels

    def apply_paper_texture(self):
        """Apply a paper texture effect to the image.

        Returns:
            PIL.Image.Image: The paper texture image.
        """
        # Create a TextureGenerator object
        texture_generator = TextureGenerator()

        # Convert the original image to a numpy array
        image_array = np.array(self.original_img_obj)

        # Generate the texture
        texture = texture_generator(
            texture_type=self.texture_type,
            texture_width=self.texture_width or image_array.shape[1],
            texture_height=self.texture_height or image_array.shape[0],
            quilt_texture=self.quilt_texture,
        )

        # Expand the texture to the specified number of channels
        if self.num_channels > 1:
            texture = np.stack([texture] * self.num_channels, axis=-1)

        # Combine the texture with the image (blending)
        combined = (image_array * self.alpha + texture * self.beta).astype(np.uint8)

        # Convert the augmented image back to a PIL image
        return Image.fromarray(combined)
