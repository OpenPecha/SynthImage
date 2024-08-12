from PIL import Image, ImageDraw, ImageFont

import SynthImage.config as config


class PageGenerator:
    def __init__(
        self,
        font_size,
        font_path,
        left_padding,
        right_padding,
        top_padding,
        bottom_padding,
        background_image=None,
    ) -> None:
        self.font_size = font_size
        self.font_path = font_path
        self.right_padding = right_padding
        self.left_padding = left_padding
        self.top_padding = top_padding
        self.bottom_padding = bottom_padding
        self.background_image = background_image
        self.extra_padding = config.EXTRA_PADDING
        self.dummy_image_mode = config.DUMMY_IMAGE_MODE
        self.dummy_image_size = config.DUMMY_IMAGE_SIZE
        self.background_color = config.BACKGROUND_COLOR
        self.text_color = config.TEXT_COLOR
        self.encoding = config.FONT_ENCODING
        self.text_bbox_x = config.TEXT_BBOX_X
        self.text_bbox_y = config.TEXT_BBOX_Y

    def get_pages(self, vol_text):
        """Segments all the page text from the volume text.

        Args:
            vol_text (str): The text of the volume.

        Returns:
            list: List of page texts.
        """
        pages = vol_text.split("\n\n")
        return pages

    def calculate_image_dimension(self, text):
        """Calculates the image dimensions.

        Args:
            text (str): Text that will be rendered into the image.

        Returns:
            tuple: Image dimensions (width, height).
        """
        dummy_image = Image.new(self.dummy_image_mode, self.dummy_image_size)
        draw = ImageDraw.Draw(dummy_image)
        font = ImageFont.truetype(
            self.font_path, self.font_size, encoding=self.encoding
        )
        lines = text.split("\n")
        horizontal_padding = self.right_padding + self.left_padding
        vertical_padding = self.top_padding + self.bottom_padding

        # Use constants for the text bounding box coordinates
        max_width = max(
            draw.textbbox((self.text_bbox_x, self.text_bbox_y), line, font=font)[2]
            for line in lines
        )
        total_height = (
            sum(
                draw.textbbox((self.text_bbox_x, self.text_bbox_y), line, font=font)[3]
                - draw.textbbox((self.text_bbox_x, self.text_bbox_y), line, font=font)[
                    1
                ]
                for line in lines
            )
            + self.extra_padding
        )
        return (
            max_width + horizontal_padding,
            total_height + vertical_padding,
        )

    def generate_page_image(self, page_text):
        """Generates a synthetic page image.

        Args:
            page_text (str): The text content to be rendered on the image.

        Returns:
            PIL.Image.Image: An Image object representing the rendered page with the provided text.
        """
        page_image_dimension = self.calculate_image_dimension(page_text)
        page_img = Image.new(
            self.dummy_image_mode, page_image_dimension, color=self.background_color
        )
        draw = ImageDraw.Draw(page_img)
        font = ImageFont.truetype(
            self.font_path, self.font_size, encoding=self.encoding
        )
        y = self.top_padding
        for line in page_text.split("\n"):
            line_bbox = draw.textbbox((self.text_bbox_x, y), line, font=font)
            draw.text((self.left_padding, y), line, font=font, fill=self.text_color)
            y += line_bbox[3] - line_bbox[1]
        page_img = page_img.convert(self.dummy_image_mode)
        return page_img
