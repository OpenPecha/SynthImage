import numpy as np
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
    ) -> None:
        self.font_size = font_size
        self.font_path = font_path
        self.right_padding = right_padding
        self.left_padding = left_padding
        self.top_padding = top_padding
        self.bottom_padding = bottom_padding

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

    def extract_lines(self, page_img, text, rotation_angle):
        if rotation_angle != 0:
            page_img = page_img.rotate(
                -rotation_angle, expand=True, fillcolor=(255, 255, 255)
            )

        lines = text.split("\n")
        y = 30
        font = ImageFont.truetype(self.font_path, self.font_size, encoding="utf-16")
        draw = ImageDraw.Draw(page_img)
        line_images = []

        # Define the amount of padding to add to the bounding box
        padding = 3  # Adjust this value as needed

        for line in lines:
            # Calculate the bounding box with padding
            line_bbox = draw.textbbox((self.left_padding, y), line, font=font)
            padded_bbox = (
                max(
                    0, line_bbox[0] - padding
                ),  # x0 - padding, ensure it doesn't go below 0
                max(
                    0, line_bbox[1] - padding
                ),  # y0 - padding, ensure it doesn't go below 0
                min(
                    page_img.width, line_bbox[2] + padding
                ),  # x1 + padding, ensure it doesn't exceed image width
                min(
                    page_img.height, line_bbox[3] + padding
                ),  # y1 + padding, ensure it doesn't exceed image height
            )

            if padded_bbox[2] > padded_bbox[0] and padded_bbox[3] > padded_bbox[1]:
                # Crop the image using the padded bounding box
                line_img = page_img.crop(
                    (padded_bbox[0], padded_bbox[1], padded_bbox[2], padded_bbox[3])
                )
                line_img = line_img.convert("RGB")
                line_img_np = np.array(line_img)
                if np.any(line_img_np != [255, 255, 255]):
                    line_images.append(line_img)

            y += (
                draw.textbbox((0, y), line, font=font)[3]
                - draw.textbbox((0, y), line, font=font)[1]
            )

        if rotation_angle != 0:
            for i in range(len(line_images)):
                line_images[i] = line_images[i].rotate(
                    rotation_angle, expand=True, fillcolor=(255, 255, 255)
                )

        return line_images

    def generate_page_image(self, page_text):
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
