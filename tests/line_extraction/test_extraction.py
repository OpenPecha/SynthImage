import tempfile
from pathlib import Path

from PIL import Image

from SynthImage.line_extraction import ExtractLines

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
