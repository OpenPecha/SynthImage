from PIL import Image, ImageDraw, ImageFont


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

    def get_pages(self, vol_text):
        """segment all the page text from the colume text

        Args:
            vol_text (str): contains the text of volume

        Returns:
            list: list of page text
        """
        pages = vol_text.split("\n\n")
        return pages

    def calculate_image_dimension(self, text):
        """Calculates the image dimension

        Args:
            text (str): text which will be in image to be generated

        Returns:
            tuple: image dimension (width,height)
        """
        dummy_image = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_image)
        font = ImageFont.truetype(self.font_path, self.font_size, encoding="utf-16")
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

    def generate_page_image(self, page_text):
        """Generates Synthetic page image

        Args:
            page_text (str): The text content to be rendered on the image

        Returns:
             PIL.Image.Image: An Image object representing the rendered page with the provided text
        """
        page_image_dimension = self.calculate_image_dimension(page_text)
        page_img = Image.new("RGB", page_image_dimension, color="white")
        draw = ImageDraw.Draw(page_img)
        font = ImageFont.truetype(self.font_path, self.font_size, encoding="utf-16")
        y = self.top_padding
        for line in page_text.split("\n"):
            line_bbox = draw.textbbox((0, y), line, font=font)
            draw.text((self.left_padding, y), line, font=font, fill=(0, 0, 0))
            y += line_bbox[3] - line_bbox[1]
        page_img = page_img.convert("RGB")
        return page_img
