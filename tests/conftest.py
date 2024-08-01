import pytest
from PIL import ImageChops


class Utils:
    @staticmethod
    def is_same_img(img1, img2):
        if img1.size != img2.size:
            return False
        diff = ImageChops.difference(img1, img2)
        if diff.getbbox():
            return False
        return True


@pytest.fixture
def utils():
    return Utils
