import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ExtractLines:
    def __init__(self, aug_img, page_text, rotation_angle, font_size, font_path):
        """Initialize the ExtractLines object.

        Args:
            aug_img (PIL.Image.Image): The augmented image from which to extract lines.
            page_text (str):The text content of the page.
            rotation_angle (int): The angle to rotate the image for line extraction.
            font_size (int): The size of the font used in the image.
            font_path (str): The path to the font file used in the image.
        """
        self.aug_img = aug_img
        self.page_text = page_text
        self.rotation_angle = rotation_angle
        self.font_size = font_size
        self.font_path = font_path

    def get_blank_img(self):
        """Create a blank white image with the same size as the augmented image.

        Returns:
            PIL.Image.Image: A blank white image.
        """
        return Image.new("RGB", self.aug_img.size, (255, 255, 255))

    def get_max_width(self, lines, blank_img, font):
        """Determine the maximum width of the lines of text and their bounding boxes.

        Args:
            lines (list of str): The lines of text.
            blank_img (_PIL.Image.Image):A blank image used for calculating bounding boxes.
            font (PIL.ImageFont.FreeTypeFont): The font used for the text.

        Returns:
            tuple: The maximum width of the lines and a list of tuples, each containing a
            line of text and its bounding box.
        """
        max_width = 0
        line_bboxes = []
        # Find the topmost position of the text after rotation
        non_white_pixels = np.where(np.array(self.aug_img) != 255)
        y = max(
            np.min(non_white_pixels[0]).item() if non_white_pixels[0].size > 0 else 0,
            30,
        )  # Start position for the first line

        for line in lines:
            # Draw the line on the blank image to calculate the bounding box
            temp_img = blank_img.copy()
            temp_draw = ImageDraw.Draw(temp_img)
            line_bbox = temp_draw.textbbox((20, y), line, font=font)

            if line_bbox[2] > line_bbox[0] and line_bbox[3] > line_bbox[1]:
                width = line_bbox[2] - line_bbox[0]
                max_width = max(max_width, width)
                line_bboxes.append((line, line_bbox))

            # Move to the next line position
            y += line_bbox[3] - line_bbox[1]
        return max_width, line_bboxes

    def get_line_images(self, max_width, line_bboxes):
        """Extract images of each line of text based on their bounding boxes.

        Args:
            max_width (int): The maximum width of the lines.
            line_bboxes (list of tuples): A list of tuples, each containing a line of text and its bounding box.
        Returns:
            list of PIL.Image.Image:  A list of images, each containing a line of text.
        """
        line_images = []
        non_white_pixels = np.where(np.array(self.aug_img) != 255)
        y = max(
            np.min(non_white_pixels[0]).item() if non_white_pixels[0].size > 0 else 0,
            30,
        )  # Start position for the first line

        for line, bbox in line_bboxes:
            # Adjust bounding box width to max_width
            x1, y1, x2, y2 = bbox
            if self.rotation_angle != 0:
                x2 = x1 + (max_width + 10)
            else:
                x2 = x1 + max_width

            # Crop the original image based on the adjusted bounding box
            line_aug_img = self.aug_img.crop((x1, y1, x2, y2))
            line_aug_img = line_aug_img.convert("RGB")
            line_aug_img_np = np.array(line_aug_img)

            # Check if there are non-white pixels
            if np.any(line_aug_img_np != [255, 255, 255]):
                line_images.append(line_aug_img)

            # Move to the next line position
            y += bbox[3] - bbox[1]
        return line_images

    def extract_lines(self):
        """Extract individual line images from the augmented image.

        Returns:
           list of PIL.Image.Image:  A list of images, each containing a line of text.
        """
        if self.rotation_angle != 0:
            self.aug_img = self.aug_img.rotate(
                -self.rotation_angle, expand=True, fillcolor=(255, 255, 255)
            )

        lines = self.page_text.split("\n")
        font = ImageFont.truetype(self.font_path, self.font_size, encoding="utf-16")

        # Create a blank image to draw text
        blank_img = self.get_blank_img()
        # Determine the maximum width
        max_width, line_bboxes = self.get_max_width(lines, blank_img, font)

        # Extract lines with the maximum width
        line_images = self.get_line_images(max_width, line_bboxes)

        if self.rotation_angle != 0:
            for i in range(len(line_images)):
                line_images[i] = line_images[i].rotate(
                    self.rotation_angle, expand=True, fillcolor=(255, 255, 255)
                )

        return line_images
