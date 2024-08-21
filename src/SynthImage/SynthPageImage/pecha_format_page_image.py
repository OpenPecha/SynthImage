from pathlib import Path

from botok import WordTokenizer
from botok.config import Config
from PIL import Image, ImageDraw, ImageFont

import SynthImage.config as config


def get_tokens(wt, text):
    tokens = wt.tokenize(text, split_affixes=False)
    return tokens


class PageGenerator:
    def __init__(
        self,
        left_padding,
        right_padding,
        top_padding,
        bottom_padding,
        chars_per_line,  # Number of characters per line
        background_image=None,
    ) -> None:
        self.right_padding = right_padding
        self.left_padding = left_padding
        self.top_padding = top_padding
        self.bottom_padding = bottom_padding
        self.background_image = background_image
        self.chars_per_line = chars_per_line
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

    def generate_page_images(
        self, vol_text, font_size, font_path, page_width, page_height
    ):
        """Generates synthetic page images with new pages starting at the first occurrence of \n\n."""
        # Fixed dimensions
        page_width = page_width
        page_height = page_height

        # Initialize list to hold all generated page images
        page_images = []

        # Split text into pages using the first occurrence of \n\n
        parts = vol_text.split("\n\n", 1)
        first_part = parts[0]
        remainder = parts[1] if len(parts) > 1 else ""

        def add_page(text_chunk):
            """Helper function to create and add a page image."""
            page_img = Image.new(
                self.dummy_image_mode,
                (page_width, page_height),
                color=self.background_color,
            )
            draw = ImageDraw.Draw(page_img)
            font = ImageFont.truetype(font_path, font_size, encoding=self.encoding)

            # Initial y-position
            y = self.top_padding

            # Tokenize text using botok
            tokens = get_tokens(self.wt, text_chunk)

            # Convert tokens to a single list of words, ignoring '\n' as line breaks
            all_words = [token.text.replace("\n", "") for token in tokens]

            # Wrap text based on the number of characters per line
            current_line = ""
            current_line_length = 0

            for word in all_words:
                # Check if adding the next word exceeds the character limit
                if current_line_length + len(word) <= self.chars_per_line:
                    current_line += word + " "
                    current_line_length += len(word) + 1  # +1 for the space
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

                    # If the text area height is exceeded, save the current page and start a new one
                    if y > page_height - self.bottom_padding:
                        page_images.append(page_img)
                        return add_page(
                            " ".join(all_words)
                        )  # Recursively handle overflow

                    # Start a new line
                    current_line = word + " "
                    current_line_length = len(word) + 1

            # Draw the final line (if any)
            if current_line:
                line_bbox = draw.textbbox(
                    (self.left_padding, y), current_line, font=font
                )
                draw.text(
                    (self.left_padding, y),
                    current_line,
                    font=font,
                    fill=self.text_color,
                )

            page_img = page_img.convert(self.dummy_image_mode)
            page_images.append(page_img)

        # Add the first part of the text
        add_page(first_part)

        # Add remaining text as additional pages
        if remainder:
            add_page(remainder)

        return page_images
