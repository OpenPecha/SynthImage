import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ExtractLines:
    def __init__(self, aug_img, page_text, rotation_angle, font_size, font_path):
        self.aug_img = aug_img
        self.page_text = page_text
        self.rotation_angle = rotation_angle
        self.font_size = font_size
        self.font_path = font_path

    def extract_lines(self):
        if self.rotation_angle != 0:
            self.aug_img = self.aug_img.rotate(
                -self.rotation_angle, expand=True, fillcolor=(255, 255, 255)
            )

        lines = self.page_text.split("\n")
        font = ImageFont.truetype(self.font_path, self.font_size, encoding="utf-16")
        line_images = []

        # Create a blank image to draw text
        blank_img = Image.new("RGB", self.aug_img.size, (255, 255, 255))

        # Step 1: Determine the maximum width
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

        # Step 2: Extract lines with the maximum width
        y = max(
            np.min(non_white_pixels[0]).item() if non_white_pixels[0].size > 0 else 0,
            30,
        )  # Reset start position for extraction
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

        if self.rotation_angle != 0:
            for i in range(len(line_images)):
                line_images[i] = line_images[i].rotate(
                    self.rotation_angle, expand=True, fillcolor=(255, 255, 255)
                )

        return line_images
