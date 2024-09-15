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

    def apply(self):
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

        # Ensure the texture is in RGB format if necessary
        if len(texture.shape) == 2:  # Grayscale texture
            texture = np.stack([texture] * 3, axis=-1)
        elif texture.shape[2] == 1:  # Single-channel texture
            texture = np.concatenate([texture] * 3, axis=-1)

        # Resize the texture to match the size of the original image
        if texture.shape[:2] != image_array.shape[:2]:
            texture = Image.fromarray(texture)
            texture = texture.resize(
                (image_array.shape[1], image_array.shape[0]), Image.BILINEAR
            )
            texture = np.array(texture)

        # Check if the original image has an alpha channel
        if image_array.shape[2] == 4:
            # If the original image has an alpha channel, we need to add an alpha channel to the texture
            if texture.shape[2] == 3:
                texture = np.concatenate(
                    [
                        texture,
                        np.ones((texture.shape[0], texture.shape[1], 1), dtype=np.uint8)
                        * 255,
                    ],
                    axis=-1,
                )

        # Expand the texture to the specified number of channels
        if texture.shape[2] != self.num_channels:
            texture = np.stack([texture] * self.num_channels, axis=-1)[
                :, :, :, : self.num_channels
            ]

        # Ensure both arrays have the same number of channels for blending
        if image_array.shape[2] != texture.shape[2]:
            raise ValueError("Number of channels in the image and texture must match")

        # Combine the texture with the image (blending)
        combined = (
            image_array[:, :, : self.num_channels] * self.alpha
            + texture[:, :, : self.num_channels] * self.beta
        ).astype(np.uint8)

        # Convert the augmented image back to a PIL image
        return Image.fromarray(combined)
