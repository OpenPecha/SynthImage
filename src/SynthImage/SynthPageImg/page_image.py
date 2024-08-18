from PIL import Image, ImageDraw, ImageFont


class PageGenerator:
    def __init__(
        self,
        left_padding,
        right_padding,
        top_padding,
        bottom_padding,
        background_image=None,
    ) -> None:
        self.right_padding = right_padding
        self.left_padding = left_padding
        self.top_padding = top_padding
        self.bottom_padding = bottom_padding
        self.background_image = background_image

    def get_pages(self, vol_text):
        """Segment the text into individual pages.

        Args:
            vol_text (str): The text of the volume.

        Returns:
            list: A list of page texts.
        """
        pages = vol_text.split("\n\n")
        return pages

    def calculate_image_dimension(self, text, font):
        """Calculates the image dimensions for a given text and font.

        Args:
            text (str): The text to be rendered in the image.
            font (ImageFont): The font object used to calculate text size.

        Returns:
            tuple: Image dimensions (width, height).
        """
        dummy_image = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_image)
        lines = text.split("\n")
        horizontal_padding = self.right_padding + self.left_padding
        vertical_padding = self.top_padding + self.bottom_padding

        max_width = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines)
        total_height = (
            sum(
                draw.textbbox((0, 0), line, font=font)[3]
                - draw.textbbox((0, 0), line, font=font)[1]
                for line in lines
            )
            + 20
        )  # Add padding
        return (
            max_width + horizontal_padding,
            total_height + vertical_padding,
        )  # Add padding

    def generate_page_image(self, page_text, font_path, font_size):
        """Generates a synthetic page image using a specific font and size.

        Args:
            page_text (str): The text content for the page.
            font_path (str): Path to the font file.
            font_size (int): Font size to be used.

        Returns:
            PIL.Image.Image: The rendered page image.
        """
        font = ImageFont.truetype(font_path, font_size, encoding="utf-16")
        page_image_dimension = self.calculate_image_dimension(page_text, font)
        page_img = Image.new("RGB", page_image_dimension, color="white")
        draw = ImageDraw.Draw(page_img)
        y = self.top_padding

        for line in page_text.split("\n"):
            line_bbox = draw.textbbox((0, y), line, font=font)
            draw.text((self.left_padding, y), line, font=font, fill=(0, 0, 0))
            y += line_bbox[3] - line_bbox[1]

        return page_img
