class RotationAugmentation:
    def __init__(self, original_img_obj, angle: float = 4):
        """Initialize the RotateAugmentation object.
        Args:
            original_img_obj (PIL.Image.Image): The input image to be augmented.
            angle (float, optional):  The angle by which to rotate the image.  Defaults to None.
        """
        self.original_img_obj = original_img_obj
        self.angle = angle

    def apply(self):
        """Apply rotation augmentation to the input image.


        Returns:
            PIL.Image.Image: The rotated image
        """
        aug_img = self.original_img_obj
        aug_img = aug_img.rotate(self.angle, expand=True, fillcolor=(255, 255, 255))
        return aug_img
