from PIL import ImageChops


def is_same_img(img1, img2):
    """Compares two images to determine if they are identical.

    Args:
        img1 (PIL.Image.Image): The first image to compare.
        img2 (PIL.Image.Image): The second image to compare.

    Returns:
        bool: `True` if the images are identical, `False` otherwise.

    """
    if img1.size != img2.size:
        return False
    diff = ImageChops.difference(img1, img2)
    if diff.getbbox():
        return False
    return True
