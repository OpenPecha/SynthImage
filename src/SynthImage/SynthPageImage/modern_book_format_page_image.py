import random
from pathlib import Path

from botok import WordTokenizer
from botok.config import Config
from PIL import Image, ImageDraw, ImageFont

import SynthImage.config as config


def get_tokens(wt, text):
    tokens = wt.tokenize(text, split_affixes=False)
    return tokens


class ModernBookPageGenerator:
    def __init__(
        self,
        left_padding,
        right_padding,
        top_padding,
        bottom_padding,
        background_image=None,
        dimensions=[
            (626, 771),
            (1063, 1536),
            (259, 194),
            (349, 522),
            (974, 1500),
            (968, 1440),
        ],  # List of possible dimensions
    ) -> None:
        self.left_padding = left_padding
        self.right_padding = right_padding
        self.top_padding = top_padding
        self.bottom_padding = bottom_padding
        self.background_image = background_image
        self.dimensions = dimensions

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
        self, vol_text, font_sizes, font_path, dimension_probs, font_size_probs
    ):
        """Generates synthetic page images with random dimensions and random font sizes."""
        page_images = []

        # Tokenize text using botok
        tokens = get_tokens(self.wt, vol_text)
        all_words = [token.text.replace("\n", "") for token in tokens]

        # Initialize variables for page drawing
        y = self.top_padding
        page_img, draw, current_line = None, None, ""

        # Counter to keep track of the number of pages for each dimension
        dimension_count = {dim: 0 for dim in self.dimensions}

        # Define tab size (number of spaces)
        tab_size = 4  # Adjust this value as needed
        # Ensure page_height is initialized before its first use
        page_height = 0  # Initialize page_height to avoid undefined errors
        current_font_size = 0  # Initialize font size to avoid undefined errors

        for word in all_words:
            # Check if a new page is needed
            if page_img is None or y > page_height - self.bottom_padding:
                if page_img is not None:
                    page_images.append(
                        (page_img, current_font_size)
                    )  # Store the font size with the image

                # Select a random dimension based on probabilities
                page_width, page_height = random.choices(
                    self.dimensions, dimension_probs
                )[0]

                # Select a random font size based on probabilities
                current_font_size = random.choices(
                    font_sizes, weights=[font_size_probs[size] for size in font_sizes]
                )[0]

                # Increment the count for the selected dimension
                dimension_count[(page_width, page_height)] += 1

                # Create a new page image
                page_img = Image.new(
                    self.dummy_image_mode,
                    (page_width, page_height),
                    color=self.background_color,
                )
                draw = ImageDraw.Draw(page_img)  # Ensure draw is initialized here
                font = ImageFont.truetype(
                    font_path, current_font_size, encoding=self.encoding
                )

                # Reset y position
                y = self.top_padding

                # Calculate maximum width allowed for text
                max_text_width = page_width - self.left_padding - self.right_padding

            # Check if the word is "༄༅།" or "༄༅༅།" and add a calculated amount of space after it
            if word == "༄༅།" or word == "༄༅༅།":
                # Add tab space (e.g., 4 spaces) after the punctuation
                word += " " * tab_size  # Adding tab_size number of spaces

            # Measure the width of the current line with the word
            if draw:  # Ensure draw is not None before calling textbbox
                word_width = draw.textbbox((0, 0), word, font=font)[2]
                line_with_word_width = draw.textbbox(
                    (0, 0), current_line + word, font=font
                )[
                    2
                ]  # Measure new line width

                # Check if adding the next word exceeds the pixel width limit
                if line_with_word_width <= max_text_width:
                    # Add word to the current line
                    current_line += word
                else:
                    # Draw the current line on the image
                    line_bbox = draw.textbbox(
                        (self.left_padding, y), current_line, font=font
                    )
                    draw.text(
                        (self.left_padding, y),
                        current_line,
                        font=font,
                        fill=self.text_color,
                    )
                    y += line_bbox[3] - line_bbox[1]

                    # Start a new line with the current word
                    current_line = word

                    # If the word itself is wider than the maximum text width, break it across lines
                    if word_width > max_text_width:
                        # Split the word by characters to fit within the max text width
                        temp_line = ""
                        for char in word:
                            char_width = draw.textbbox(
                                (0, 0), temp_line + char, font=font
                            )[2]
                            if char_width <= max_text_width:
                                temp_line += char
                            else:
                                # Draw the split line
                                line_bbox = draw.textbbox(
                                    (self.left_padding, y), temp_line, font=font
                                )
                                draw.text(
                                    (self.left_padding, y),
                                    temp_line,
                                    font=font,
                                    fill=self.text_color,
                                )
                                y += line_bbox[3] - line_bbox[1]

                                # Start new line with the remaining characters
                                temp_line = char
                        # Draw the remaining characters
                        current_line = temp_line

        # Draw the final line (if any)
        if current_line and draw:
            line_bbox = draw.textbbox((self.left_padding, y), current_line, font=font)
            draw.text(
                (self.left_padding, y), current_line, font=font, fill=self.text_color
            )

        # Add the final page image to the list
        if page_img is not None:
            page_images.append((page_img, current_font_size))

        return page_images, dimension_count
