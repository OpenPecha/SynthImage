from pathlib import Path

import cv2
import numpy as np

from SynthImage.Augmentation.background import BackgroundAugmentation
from SynthImage.SynthPageImage.pecha_format_page_image import (
    PechaPageGenerator,
    draw_tight_line_bounding_boxes,
)

font_path = Path("./data/fonts/Drutsa_short/Kangba Derchi-Drutsa.ttf")
background_folder = "./data/backgrounds"
pgobject = PechaPageGenerator(50, 50, 45, 45, [(1123, 265), (794, 265)])
text = """  ༄༅༅། །རྒྱ་གར་སྐད་དུ། བི་ན་ཡ་བསྟུ། བོད་སྐད་དུ། འདུལ་བ་གཞི། བམ་པོ་དང་པོ། དཀོན་མཆོག་གསུམ་ལ་ཕྱག་འཚལ་ལོ། །གང་གིས་འཆིང་
རྣམས་ཡང་དག་རབ་བཅད་ཅིང་། །མུ་སྟེགས་ཚོགས་རྣམས་ཐམས་ཅད་རབ་བཅོམ་སྟེ། །སྡེ་དང་བཅས་པའི་བདུད་རྣམས་ངེས་བཅོམ་ནས། །བྱང་ཆུབ་འདི་བརྙེས་དེ་ལ་
ཕྱག་འཚལ་ལོ། །ཁྱིམ་དོན་ཆེ་ཆུང་སྤངས་ཏེ་དང་པོར་རབ་འབྱུང་དཀའ། །རབ་བྱུང་ཐོབ་ནས་ཡུལ་སྤྱད་དག་གིས་དགའ་ཐོབ་དཀའ། །མངོན་དགའ་ཇི་བཞིན་དོན་བསྐྱེད་ཡང་
དག་བྱེད་པ་དཀའ། །ངུར་སྨྲིག་གོས་འཆང་མཁས་པ་ཚུལ་ལས་ཉམས་པ་དཀའ། །གཞི་རྣམས་ཀྱི་སྤྱི་སྡོམ་ལ། རབ་འབྱུང་གསོ་སྦྱོང་གཞི་དང་ནི། །དགག་དབྱེ་དབྱར་དང་ཀོ་
ལྤགས་གཞི། །སྨན་དང་གོས་དང་སྲ་བརྐྱང་དང་། །ཀཽ་ཤཱམ་བཱི་དང་ལས་ཀྱི་གཞི། །དམར་སེར་ཅན་དང་གང་ཟག་དང་། །སྤོ་དང་གསོ་སྦྱོང་བཞག་པ་དང་། །གནས་མལ་དང་ནི་

རྩོད་པ་དང་། །དགེ་འདུན་དབྱེན་རྣམས་བསྡུས་པ་ཡིན། །རབ་ཏུ་འབྱུང་བའི་གཞིའི་སྤྱི་སྡོམ་ལ། ཤཱ་རིའི་བུ་དང་མུ་སྟེགས་ཅན། །དགེ་ཚུལ་གཉིས་དང་བྱ་རོག་སྐྲོད། །དགྲ་བཅོམ་བསད་དང་ལག་རྡུམ་གྱི། །སྡེ་ཚན་
ཡང་དག་བསྡུས་པ་ཡིན། །སྡོམ་ལ། ཤཱ་རིའི་བུ་དང་རབ་འབྱུང་དང་། །བསྙེན་པར་རྫོགས་པར་གནང་བ་དང་། །ཉེ་སྡེས་ཚོགས་ནི་བསྡུས་པ་དང་། །ལྔ་པའི་སྡེ་ཚན་བསྡུས་པ་ཡིན། །བྱང་ཆུབ་སེམས་དཔའ་དགའ་ལྡན་གྱི་གནས་ན་བཞུགས་པ་ན།
ཡུལ་ཨང་ག་དག་ན་ཨང་གའི་རྒྱལ་པོ་ཞེས་བྱ་བས་རྒྱལ་སྲིད་འབྱོར་པ། རྒྱས་པ་བདེ་བ་ལོ་ལེགས་པ་སྐྱེ་བོ་དང་མི་མང་པོས་གང་བ་བྱེད་དུ་བཅུག་གོ། །ཡུལ་མ་ག་དྷཱ་དག་ན་ཡང་རྒྱལ་པོ་པད་མ་ཆེན་པོ་ཞེས་བྱ་བས། རྒྱལ་སྲིད་འབྱོར་པ་རྒྱས་པ་བདེ་བ་ལོ་
ལེགས་པ་སྐྱེ་བོ་དང་མི་མང་པོས་གང་བ་བྱེད་དུ་བཅུག་གོ། །རེས་འགའ་ནི་ཨང་གའི་རྒྱལ་པོ་དཔུང་དང་མཐུ་ཆེ་བ་ཡིན་ལ། རེས་འགའ་ནི་རྒྱལ་པོ་པད་མ་ཆེན་པོ་དཔུང་དང་མཐུ་ཆེ་བ་ཡིན་ནོ། །གང་གི་ཚེ་ཨང་གའི་རྒྱལ་པོ་དཔུང་དང་མཐུ་ཆེ་བ་དེའི་ཚེ་
ན། དེས་དཔུང་གི་ཚོགས་ཡན་ལག་བཞི་པ། གླང་པོ་ཆེ་པའི་ཚོགས་དང་། རྟ་པའི་ཚོགས་དང་། ཤིང་རྟ་པའི་ཚོགས་དང་། དཔུང་བུ་ཆུང་གི་ཚོགས་གོ་བསྐོན་ཏེ། ཡུལ་མ་ག་དྷཱ་རྒྱལ་པོའི་ཁབ་མ་གཏོགས་པ་བཅོམ་ནས་ཕྱིར་ལྡོག་པར་
བྱེད་དོ། །གང་གི་ཚེ་རྒྱལ་པོ་པད་མ་ཆེན་པོ་དཔུང་དང་མཐུ་ཆེ་བ་དེའི་ཚེ་ན། དེས་ཀྱང་དཔུང་གི་ཚོགས་ཡན་ལག་བཞི་པ། གླང་པོ་ཆེ་པའི་ཚོགས་དང་། རྟ་པའི་ཚོགས་དང་། ཤིང་རྟ་པའི་ཚོགས་དང་། དཔུང་བུ་ཆུང་གི་ཚོགས་གོ་བསྐོན་ཏེ།

རྩོད་པ་དང་། །དགེ་འདུན་དབྱེན་རྣམས་བསྡུས་པ་ཡིན། །རབ་ཏུ་འབྱུང་བའི་གཞིའི་སྤྱི་སྡོམ་ལ། ཤཱ་རིའི་བུ་དང་མུ་སྟེགས་ཅན། །དགེ་ཚུལ་གཉིས་དང་བྱ་རོག་སྐྲོད། །དགྲ་བཅོམ་བསད་དང་ལག་རྡུམ་གྱི། །སྡེ་ཚན་
ཡང་དག་བསྡུས་པ་ཡིན། །སྡོམ་ལ། ཤཱ་རིའི་བུ་དང་རབ་འབྱུང་དང་། །བསྙེན་པར་རྫོགས་པར་གནང་བ་དང་། །ཉེ་སྡེས་ཚོགས་ནི་བསྡུས་པ་དང་། །ལྔ་པའི་སྡེ་ཚན་བསྡུས་པ་ཡིན། །བྱང་ཆུབ་སེམས་དཔའ་དགའ་ལྡན་གྱི་གནས་ན་བཞུགས་པ་ན།
ཡུལ་ཨང་ག་དག་ན་ཨང་གའི་རྒྱལ་པོ་ཞེས་བྱ་བས་རྒྱལ་སྲིད་འབྱོར་པ། རྒྱས་པ་བདེ་བ་ལོ་ལེགས་པ་སྐྱེ་བོ་དང་མི་མང་པོས་གང་བ་བྱེད་དུ་བཅུག་གོ། །ཡུལ་མ་ག་དྷཱ་དག་ན་ཡང་རྒྱལ་པོ་པད་མ་ཆེན་པོ་ཞེས་བྱ་བས། རྒྱལ་སྲིད་འབྱོར་པ་རྒྱས་པ་བདེ་བ་ལོ་
ལེགས་པ་སྐྱེ་བོ་དང་མི་མང་པོས་གང་བ་བྱེད་དུ་བཅུག་གོ། །རེས་འགའ་ནི་ཨང་གའི་རྒྱལ་པོ་དཔུང་དང་མཐུ་ཆེ་བ་ཡིན་ལ། རེས་འགའ་ནི་རྒྱལ་པོ་པད་མ་ཆེན་པོ་དཔུང་དང་མཐུ་ཆེ་བ་ཡིན་ནོ། །གང་གི་ཚེ་ཨང་གའི་རྒྱལ་པོ་དཔུང་དང་མཐུ་ཆེ་བ་དེའི་ཚེ་
ན། དེས་དཔུང་གི་ཚོགས་ཡན་ལག་བཞི་པ། གླང་པོ་ཆེ་པའི་ཚོགས་དང་། རྟ་པའི་ཚོགས་དང་། ཤིང་རྟ་པའི་ཚོགས་དང་། དཔུང་བུ་ཆུང་གི་ཚོགས་གོ་བསྐོན་ཏེ། ཡུལ་མ་ག་དྷཱ་རྒྱལ་པོའི་ཁབ་མ་གཏོགས་པ་བཅོམ་ནས་ཕྱིར་ལྡོག་པར་
བྱེད་དོ། །གང་གི་ཚེ་རྒྱལ་པོ་པད་མ་ཆེན་པོ་དཔུང་དང་མཐུ་ཆེ་བ་དེའི་ཚེ་ན། དེས་ཀྱང་དཔུང་གི་ཚོགས་ཡན་ལག་བཞི་པ། གླང་པོ་ཆེ་པའི་ཚོགས་དང་། རྟ་པའི་ཚོགས་དང་། ཤིང་རྟ་པའི་ཚོགས་དང་། དཔུང་བུ་ཆུང་གི་ཚོགས་གོ་བསྐོན་ཏེ།
"""  # noqa
font_sizes = [30, 11]  # Adjust the font size as needed


# Define probabilities for each dimension
dimension_probs = [0.5, 0.5]

# Define probabilities for each font size
font_size_probs = {30: 0, 11: 1}


def test_generate_pecha_page_images():
    (
        page_images_with_bbox,
        page_images_without_bbox,
        dimension_count,
        bbox_details,
        polygon_bbox_details,
    ) = pgobject.generate_pecha_page_images(
        text, font_sizes, font_path, dimension_probs, font_size_probs
    )
    # Get the dimension of the current page image
    page_width, page_height = page_images_with_bbox[0][0].size
    # Create the filename with dimension, font size, and font name
    dimension_prefix = f"{page_width}x{page_height}"
    page_images_dir = Path("./tests/data/pecha_format_page_img")
    page_images_dir.mkdir(parents=True, exist_ok=True)
    page_image_path = (
        page_images_dir / f"{dimension_prefix}_pecha_format_page_image.png"
    )
    page_images_with_bbox[0][0].save(page_image_path)

    # Save the image without bounding boxes
    page_image_without_bbox_path = (
        page_images_dir / f"{dimension_prefix}_pecha_format_page_image_without_bbox.png"
    )
    page_images_without_bbox[0][0].save(page_image_without_bbox_path)

    rotation_aug = BackgroundAugmentation(
        page_images_with_bbox[0][0], "./data/backgrounds"
    )  # Use a fixed angle or generate a random angle within a range
    rotated_image = rotation_aug.apply()
    # Save the rotated image with bounding boxes
    rotated_image_path = page_images_dir / f"{dimension_prefix}_aug_withb_box.png"
    cv2.imwrite(
        str(rotated_image_path),
        cv2.cvtColor(np.array(rotated_image), cv2.COLOR_RGB2BGR),
    )


def test_draw_tight_line_bounding_boxes():
    # Generate a pecha format page image
    (
        page_images_with_bbox,
        page_images_without_bbox,
        _,
        bbox_details,
        polygon_bbox_details,
    ) = pgobject.generate_pecha_page_images(
        text, font_sizes, font_path, dimension_probs, font_size_probs
    )
    sample_image = np.array(page_images_without_bbox[0][0])

    # Call the function
    (
        image_with_bbox,
        image_without_bbox,
        bbox_details_new,
        polygon_bbox_details_new,
    ) = draw_tight_line_bounding_boxes(sample_image, "test_image.png", 1)

    # Save the output images for visual inspection
    output_dir = Path("./tests/data/pecha_format_page_img")
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "test_image_with_bbox.png"), image_with_bbox)
    cv2.imwrite(str(output_dir / "test_image_without_bbox.png"), image_without_bbox)
