import tempfile
from pathlib import Path

from PIL import Image, ImageFont

from SynthImage.LineExtraction.line_extraction import ExtractLines

original_img_path = Path(
    "./tests/augmentation/data/expected_rotate_image/expected_rotate_image_3.png"
)
font_path = "./tests/font/monlam_uni_ochan1.ttf"
original_img_obj = Image.open(original_img_path)

text = """  ༄༅༅། །རྒྱ་གར་སྐད་དུ། བི་ན་ཡ་བསྟུ། བོད་སྐད་དུ། འདུལ་བ་གཞི། བམ་པོ་དང་པོ། དཀོན་མཆོག་གསུམ་ལ་ཕྱག་འཚལ་ལོ། །གང་གིས་འཆིང་
            རྣམས་ཡང་དག་རབ་བཅད་ཅིང་། །མུ་སྟེགས་ཚོགས་རྣམས་ཐམས་ཅད་རབ་བཅོམ་སྟེ། །སྡེ་དང་བཅས་པའི་བདུད་རྣམས་ངེས་བཅོམ་ནས། །བྱང་ཆུབ་འདི་བརྙེས་དེ་ལ་
            ཕྱག་འཚལ་ལོ། །ཁྱིམ་དོན་ཆེ་ཆུང་སྤངས་ཏེ་དང་པོར་རབ་འབྱུང་དཀའ། །རབ་བྱུང་ཐོབ་ནས་ཡུལ་སྤྱད་དག་གིས་དགའ་ཐོབ་དཀའ། །མངོན་དགའ་ཇི་བཞིན་དོན་བསྐྱེད་ཡང་
            དག་བྱེད་པ་དཀའ། །ངུར་སྨྲིག་གོས་འཆང་མཁས་པ་ཚུལ་ལས་ཉམས་པ་དཀའ། །གཞི་རྣམས་ཀྱི་སྤྱི་སྡོམ་ལ། རབ་འབྱུང་གསོ་སྦྱོང་གཞི་དང་ནི། །དགག་དབྱེ་དབྱར་དང་ཀོ་
            ལྤགས་གཞི། །སྨན་དང་གོས་དང་སྲ་བརྐྱང་དང་། །ཀཽ་ཤཱམ་བཱི་དང་ལས་ཀྱི་གཞི། །དམར་སེར་ཅན་དང་གང་ཟག་དང་། །སྤོ་དང་གསོ་སྦྱོང་བཞག་པ་དང་། །གནས་མལ་དང་ནི་"""  # noqa

extractObject = ExtractLines(original_img_obj, text, 3, 30, font_path)


def test_line_extraction(utils):
    """Test the line extraction process by comparing actual extracted lines with expected results.

    Args:
        utils: A utility object that contains helper functions for testing,
               specifically an `is_same_img` function that compares two images.
    """
    line_images = extractObject.extract_lines()
    expected_extracted_lines_dir = Path(
        "./tests/line_extraction/data/expected_line_extraction"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        for idx, line_img in enumerate(line_images):
            actual_line_images_path = Path(tempdirname) / f"actual_line_images{idx}.png"
            line_img.save(actual_line_images_path)
            expected_extracted_lines_path = (
                Path(expected_extracted_lines_dir)
                / f"expected_line_extraction{idx}.png"
            )
            expected_line_image = Image.open(expected_extracted_lines_path)
            actual_line_image = Image.open(actual_line_images_path)
            assert utils.is_same_img(expected_line_image, actual_line_image)


def test_blank_image(utils):
    """Test the creation of a blank image.

    Args:
        utils: A utility object that contains helper functions for testing,
               specifically an `is_same_img` function that compares two images.
    """
    blank_img = extractObject.get_blank_img()
    expected_blank_image_path = Path(
        "./tests/line_extraction/data/expected_blank_image/expected_blank_image.png"
    )
    with tempfile.TemporaryDirectory() as tempdirname:
        actual_blank_image_path = Path(tempdirname) / "actual_blank_image.png"
        blank_img.save(actual_blank_image_path)
        expected_blank_image = Image.open(expected_blank_image_path)
        actual_blank_image = Image.open(actual_blank_image_path)
        assert utils.is_same_img(expected_blank_image, actual_blank_image)


def test_max_width():
    """Test the calculation of the maximum width of the lines."""
    blank_img = extractObject.get_blank_img()
    font = ImageFont.truetype(font_path, 30)
    lines = text.split("\n")
    max_width, line_bboxes = extractObject.get_max_width(lines, blank_img, font)
    assert max_width == 1169


def test_line_images(utils):
    """Test the creation of line images.

    Args:
        utils: A utility object that contains helper functions for testing,
               specifically an `is_same_img` function that compares two images.
    """
    blank_img = extractObject.get_blank_img()
    font = ImageFont.truetype(font_path, 30)
    lines = text.split("\n")
    max_width, line_bboxes = extractObject.get_max_width(lines, blank_img, font)
    line_images = extractObject.get_line_images(max_width, line_bboxes)
    expected_line_images_dir = Path("./tests/line_extraction/data/expected_line_images")
    with tempfile.TemporaryDirectory() as tempdirname:

        for idx, line_image in enumerate(line_images):
            actual_line_images_path = Path(tempdirname) / f"actual_line_image{idx}.png"
            line_image.save(actual_line_images_path)
            expected_line_images_path = (
                expected_line_images_dir / f"expected_line_image{idx}.png"
            )
            expected_line_image = Image.open(expected_line_images_path)
            actual_line_image = Image.open(actual_line_images_path)
            assert utils.is_same_img(actual_line_image, expected_line_image)
