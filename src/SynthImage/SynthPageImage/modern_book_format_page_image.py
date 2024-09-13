import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from botok import WordTokenizer
from botok.config import Config
from PIL import Image, ImageDraw, ImageFont

import SynthImage.config as config


def get_tokens(wt, text):
    """
    Tokenize the given text using the provided WordTokenizer.

    Args:
        wt (WordTokenizer): An instance of WordTokenizer.
        text (str): The text to be tokenized.

    Returns:
        list: A list of tokens.
    """
    tokens = wt.tokenize(text, split_affixes=False)
    return tokens


def draw_tight_line_bounding_boxes(
    image: np.ndarray, filename: str, page_number: int
) -> tuple:
    """
    Draw tight bounding boxes around text lines in the image.

    Args:
        image (np.ndarray): The input image.
        filename (str): The name of the file being processed.
        page_number (int): The current page number.

    Returns:
        tuple: A tuple containing:
            - image_with_bbox (np.ndarray): The image with bounding boxes drawn.
            - image_without_bbox (np.ndarray): The original image without bounding boxes.
            - bbox_details (list): A list of dictionaries containing bounding box details.
            - polygon_bbox_details (list): A list of dictionaries containing polygon bounding box details.
    """
    bbox_details = []
    polygon_bbox_details = []

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (27, 1))
    dilated_image = cv2.dilate(binary_image, kernel, iterations=2)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=1)
    contours, _ = cv2.findContours(
        eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    bbox_idx = 0
    polygon_idx = 0

    image_with_bbox = image.copy()
    image_without_bbox = image.copy()

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:
            epsilon = 0.0005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            x, y, w, h = cv2.boundingRect(approx)
            center = [x + w / 2, y + h / 2]
            points = [[x, y], [x, y + h], [x + w, y + h], [x + w, y]]

            bbox_details.append(
                {
                    "id": f"{filename}_bbox_{bbox_idx}",
                    "image": f"https://s3.amazonaws.com/monlam.ai.ocr/line_segmentations/Images/{filename}_bbox_{bbox_idx}",  # noqa
                    "height": h,
                    "width": w,
                    "center": center,
                    "points": points,
                    "page_number": page_number,
                }
            )
            bbox_idx += 1

            polygon_points = approx.reshape(-1, 2).tolist()
            polygon_bbox_details.append(
                {
                    "id": f"{filename}_polygon_bbox_{polygon_idx}",
                    "image": f"https://s3.amazonaws.com/monlam.ai.ocr/line_segmentations/Images/{filename}_polygon_bbox_{polygon_idx}",  # noqa
                    "points": polygon_points,
                    "page_number": page_number,
                }
            )
            polygon_idx += 1

            cv2.polylines(
                image_with_bbox, [approx], isClosed=True, color=(0, 255, 0), thickness=1
            )

    return image_with_bbox, image_without_bbox, bbox_details, polygon_bbox_details


class ModernBookPageGenerator:
    """
    A class for generating modern book page images with text.

    Attributes:
        left_padding (int): Left padding of the text area.
        right_padding (int): Right padding of the text area.
        top_padding (int): Top padding of the text area.
        bottom_padding (int): Bottom padding of the text area.
        dimensions (list): List of possible page dimensions.
        line_spacing (int): Spacing between lines of text.
    """

    def __init__(
        self,
        left_padding,
        right_padding,
        top_padding,
        bottom_padding,
        dimensions=[
            (626, 771),
            (1063, 1536),
            (259, 194),
            (349, 522),
            (974, 1500),
            (968, 1440),
        ],
        line_spacing=10,
    ) -> None:
        """
        Initialize the ModernBookPageGenerator.

        Args:
            left_padding (int): Left padding of the text area.
            right_padding (int): Right padding of the text area.
            top_padding (int): Top padding of the text area.
            bottom_padding (int): Bottom padding of the text area.
            dimensions (list): List of possible page dimensions.
            line_spacing (int): Spacing between lines of text.
        """
        self.left_padding = left_padding
        self.right_padding = right_padding
        self.top_padding = top_padding
        self.bottom_padding = bottom_padding
        self.dimensions = dimensions
        self.line_spacing = line_spacing

        # Load configuration settings
        self.extra_padding = config.EXTRA_PADDING
        self.dummy_image_mode = config.DUMMY_IMAGE_MODE
        self.dummy_image_size = config.DUMMY_IMAGE_SIZE
        self.background_color = config.BACKGROUND_COLOR
        self.text_color = config.TEXT_COLOR
        self.encoding = config.FONT_ENCODING
        self.text_bbox_x = config.TEXT_BBOX_X
        self.text_bbox_y = config.TEXT_BBOX_Y

        # Initialize WordTokenizer
        self.config = Config(dialect_name="general", base_path=Path.home())
        self.wt = WordTokenizer(config=self.config)

    def generate_modern_page_images(
        self,
        vol_text: str,
        font_sizes: List[int],
        font_path: str,
        dimension_probs: List[float],
        font_size_probs: Dict[int, float],
    ) -> Tuple[
        List[Tuple[Image.Image, int]],
        List[Tuple[Image.Image, int]],
        Dict[str, int],
        List[Dict],
        List[Dict],
    ]:
        """
        Generate modern book page images with text.

        Args:
            vol_text (str): The text to be rendered on the pages.
            font_sizes (list): List of possible font sizes.
            font_path (str): Path to the font file.
            dimension_probs (list): Probabilities for selecting page dimensions.
            font_size_probs (dict): Probabilities for selecting font sizes.

        Returns:
            tuple: A tuple containing:
                - page_images_with_bbox (list): List of tuples (image with bounding boxes, font size).
                - page_images_without_bbox (list): List of tuples (image without bounding boxes, font size).
                - dimension_counter (dict): Counter for page dimensions used.
                - bbox_details (list): List of bounding box details for all pages.
                - polygon_bbox_details (list): List of polygon bounding box details for all pages.
        """
        page_images_with_bbox = []
        page_images_without_bbox = []
        bbox_details = []
        polygon_bbox_details = []

        tokens = get_tokens(self.wt, vol_text)
        all_words = [token.text.replace("\n", "") for token in tokens]

        y = self.top_padding
        page_img, draw, current_line = None, None, ""

        dimension_counter: Dict[str, int] = {}  # Add type annotation here
        tab_size = 4
        page_height = 0
        page_width = 0
        page_number = 1
        current_font_size = 0

        for word in all_words:
            if (
                page_img is None
                or y > page_height - self.bottom_padding - current_font_size
            ):
                if page_img is not None:
                    page_img_np = np.array(page_img)

                    dimension_prefix = f"{page_width}x{page_height}"
                    dimension_counter[dimension_prefix] = (
                        dimension_counter.get(dimension_prefix, 0) + 1
                    )
                    count = dimension_counter[dimension_prefix]
                    filename = f"page_{page_number}_{dimension_prefix}_count_{count}_font{current_font_size}_{Path(font_path).stem}.png"  # noqa

                    (
                        page_img_with_bbox,
                        page_img_without_bbox,
                        bbox_details_page,
                        polygon_bbox_details_page,
                    ) = draw_tight_line_bounding_boxes(
                        page_img_np, filename, page_number
                    )
                    page_img_with_bbox = Image.fromarray(page_img_with_bbox)
                    page_img_without_bbox = Image.fromarray(page_img_without_bbox)

                    page_images_with_bbox.append(
                        (page_img_with_bbox, current_font_size)
                    )
                    page_images_without_bbox.append(
                        (page_img_without_bbox, current_font_size)
                    )
                    bbox_details.extend(bbox_details_page)
                    polygon_bbox_details.extend(polygon_bbox_details_page)
                    page_number += 1

                page_width, page_height = random.choices(
                    self.dimensions, dimension_probs
                )[0]
                current_font_size = random.choices(
                    font_sizes, weights=[font_size_probs[size] for size in font_sizes]
                )[0]

                page_img = Image.new(
                    "RGB", (page_width, page_height), color=self.background_color
                )
                draw = ImageDraw.Draw(page_img)
                font = ImageFont.truetype(
                    font_path, current_font_size, encoding=self.encoding
                )
                y = self.top_padding
                max_text_width = page_width - self.left_padding - self.right_padding

            if "༄༅།" in word or "༄༅༅།" in word:
                word = word.replace("༄༅།", "༄༅།" + " " * tab_size)
                word = word.replace("༄༅༅།", "༄༅༅།" + " " * tab_size)

            if draw:
                word_width = draw.textbbox((0, 0), word, font=font)[2]
                line_with_word_width = draw.textbbox(
                    (0, 0), current_line + word, font=font
                )[2]

                if line_with_word_width <= max_text_width:
                    current_line += word
                else:
                    line_bbox = draw.textbbox(
                        (self.left_padding, y), current_line, font=font
                    )
                    draw.text(
                        (self.left_padding, y),
                        current_line,
                        font=font,
                        fill=self.text_color,
                    )
                    y += line_bbox[3] - line_bbox[1] + self.line_spacing
                    current_line = word

                    if word_width > max_text_width:
                        temp_line = ""
                        for char in word:
                            char_width = draw.textbbox(
                                (0, 0), temp_line + char, font=font
                            )[2]
                            if char_width <= max_text_width:
                                temp_line += char
                            else:
                                line_bbox = draw.textbbox(
                                    (self.left_padding, y), temp_line, font=font
                                )
                                draw.text(
                                    (self.left_padding, y),
                                    temp_line,
                                    font=font,
                                    fill=self.text_color,
                                )
                                y += line_bbox[3] - line_bbox[1] + self.line_spacing
                                temp_line = char
                        current_line = temp_line

        if current_line and draw:
            line_bbox = draw.textbbox((self.left_padding, y), current_line, font=font)
            draw.text(
                (self.left_padding, y), current_line, font=font, fill=self.text_color
            )

            page_img_np = np.array(page_img)
            dimension_prefix = f"{page_width}x{page_height}"
            dimension_counter[dimension_prefix] = (
                dimension_counter.get(dimension_prefix, 0) + 1
            )
            count = dimension_counter[dimension_prefix]
            filename = f"page_{page_number}_{dimension_prefix}_count_{count}_font{current_font_size}_{Path(font_path).stem}.png"  # noqa

            (
                page_img_with_bbox,
                page_img_without_bbox,
                bbox_details_page,
                polygon_bbox_details_page,
            ) = draw_tight_line_bounding_boxes(page_img_np, filename, page_number)
            page_img_with_bbox = Image.fromarray(page_img_with_bbox)
            page_img_without_bbox = Image.fromarray(page_img_without_bbox)

            page_images_with_bbox.append((page_img_with_bbox, current_font_size))
            page_images_without_bbox.append((page_img_without_bbox, current_font_size))
            bbox_details.extend(bbox_details_page)
            polygon_bbox_details.extend(polygon_bbox_details_page)

        return (
            page_images_with_bbox,
            page_images_without_bbox,
            dimension_counter,
            bbox_details,
            polygon_bbox_details,
        )
