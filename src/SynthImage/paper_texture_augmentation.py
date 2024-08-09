import numpy as np
from augraphy import TextureGenerator
from PIL import Image


class PaperTextureAugmentation:
    def __init__(self, original_img_obj):
        """Initialize the PaperTextureAugmentation class with an image.

        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
        """
        self.original_img_obj = original_img_obj

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
            texture_type="rough_stains",
            texture_width=image_array.shape[1],
            texture_height=image_array.shape[0],
            quilt_texture=0,
        )

        # Expand the texture to 3 channels to match the RGB image
        texture = np.stack([texture] * 3, axis=-1)

        # Combine the texture with the image (blending)
        combined = (image_array * 0.7 + texture * 0.3).astype(np.uint8)

        # Convert the augmented image back to a PIL image
        return Image.fromarray(combined)
