import math
import random
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from SynthImage.page_image import PageGenerator

font_dir = Path(__file__).parent / "../.." / "data" / "fonts"
# Recursively get all font files from the subdirectories in the fonts directory
font_files = [
    font_file
    for font_folder in font_dir.iterdir()
    if font_folder.is_dir()
    for font_file in font_folder.glob("*.ttf")
]
if not font_files:
    raise FileNotFoundError("No font files found in the specified directory.")
# Randomly select a font from the list of font files
selected_font = random.choice(font_files)
# Randomize the font size, padding values within the given ranges
font_size = random.randint(20, 30)


pgobject = PageGenerator(30, str(selected_font), 10, 10, 30, 30)


class DistortionMode(Enum):
    """A simple selection for the mode used for font contour distortion"""

    additive = 0
    subtractive = 1


def distort_line(
    image: np.ndarray,
    mode: DistortionMode = DistortionMode.additive,
    edge_tresh1: int = 100,
    edge_tresh2: int = 200,
    kernel_width: int = 2,
    kernel_height: int = 1,
    kernel_iterations=2,
):
    if type(image) is not np.array:
        image = np.array(image)
    edges = cv2.Canny(image, edge_tresh1, edge_tresh2)
    if edges is None:
        return image
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    kernel = np.ones((kernel_width, kernel_height), np.uint8)
    edges = cv2.erode(edges, kernel, iterations=kernel_iterations)
    edges = cv2.dilate(edges, kernel, iterations=kernel_iterations)
    indices = np.where(edges[:, :] == 255)
    cv_image_added: np.ndarray = image.copy()
    if mode == DistortionMode.additive:
        cv_image_added[indices[0], indices[1], :] = [0]
    else:
        cv_image_added[indices[0], indices[1], :] = [255]
    return cv_image_added


class WaveDeformer:
    def __init__(self, grid: int = 20, multiplier: int = 6, offset: int = 70) -> None:
        self.multiplier = multiplier
        self.offset = offset
        self.grid = grid

    def transform(self, x, y):
        y = y + self.multiplier * math.sin(x / self.offset)
        return x, y

    def transform_rectangle(self, x0, y0, x1, y1):
        return (
            *self.transform(x0, y0),
            *self.transform(x0, y1),
            *self.transform(x1, y1),
            *self.transform(x1, y0),
        )

    def getmesh(self, img):
        self.w, self.h = img.size
        gridspace = self.grid
        target_grid = []
        for x in range(0, self.w, gridspace):
            for y in range(0, self.h, gridspace):
                target_grid.append((x, y, x + gridspace, y + gridspace))
        source_grid = [self.transform_rectangle(*rect) for rect in target_grid]
        return [t for t in zip(target_grid, source_grid)]


def deform_image(image: np.ndarray):
    grid_size = random.randint(1, 140)
    multiplier = random.randint(1, 3)
    offset = random.randint(1, 100)
    deformer = WaveDeformer(grid=grid_size, multiplier=multiplier, offset=offset)
    pil_image = Image.fromarray(image)
    deformed_img = ImageOps.deform(pil_image, deformer)
    return np.array(deformed_img)


def create_realistic_torn_mask(width, height, num_tears=5, tear_size=30):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for _ in range(num_tears):
        edge = random.choice(["top", "bottom", "left", "right"])

        if edge == "top":
            x0 = random.randint(0, width - tear_size)
            y0 = 0
            jagged_line = [
                (x0 + i + random.randint(-5, 5), y0 + random.randint(0, 5))
                for i in range(0, tear_size, 5)
            ]
        elif edge == "bottom":
            x0 = random.randint(0, width - tear_size)
            y0 = height
            jagged_line = [
                (x0 + i + random.randint(-5, 5), y0 - random.randint(0, 5))
                for i in range(0, tear_size, 5)
            ]
        elif edge == "left":
            y0 = random.randint(0, height - tear_size)
            x0 = 0
            jagged_line = [
                (x0 + random.randint(0, 5), y0 + i + random.randint(-5, 5))
                for i in range(0, tear_size, 5)
            ]
        elif edge == "right":
            y0 = random.randint(0, height - tear_size)
            x0 = width
            jagged_line = [
                (x0 - random.randint(0, 5), y0 + i + random.randint(-5, 5))
                for i in range(0, tear_size, 5)
            ]

        draw.line(jagged_line, fill=255, width=tear_size)

    return mask


def apply_realistic_torn_effect(image: Image.Image, num_tears=5, tear_size=30):
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    width, height = image.size
    mask = create_realistic_torn_mask(width, height, num_tears, tear_size)

    torn_image = image.copy()
    torn_image.paste((0, 0, 0, 0), mask=mask)

    return torn_image


class PageAugmentationGenerator:
    def __init__(self, text_folder, output_dir) -> None:
        self.text_folder = text_folder
        self.output_dir = output_dir

    def apply_background(self, page_img):
        """
        Applies a background to the synthetic page image if no background is already present.
        """
        if page_img.mode != "RGBA":
            page_img = page_img.convert("RGBA")
        background_dir = (
            Path(__file__).parent / "../.." / "background"
        )  # Update this path
        background_files = list(background_dir.glob("*.jpg"))
        if not background_files:
            return page_img
        # Randomly select a background image
        background_file = random.choice(background_files)
        background = Image.open(background_file).convert("RGBA")
        # Resize background to match the synthetic page image
        background = background.resize(page_img.size, Image.Resampling.LANCZOS)
        # Create a new image with a white background and the synthetic image on top
        combined = Image.new("RGBA", page_img.size, (255, 255, 255, 255))
        combined.paste(background, (0, 0), background)
        combined.paste(page_img, (0, 0), page_img)
        return combined.convert("RGB")  # Convert back to RGB if needed

    def apply_augmentation(self, page_img):
        possible_augmentations = [
            ("brightness", lambda page_img: self.augment_brightness(page_img)),
            ("contrast", lambda page_img: self.augment_contrast(page_img)),
            ("sharpness", lambda page_img: self.augment_sharpness(page_img)),
            ("rotate", lambda page_img: self.augment_rotate(page_img)),
            ("distort", lambda page_img: self.augment_distort(page_img)),
            ("deform", lambda page_img: self.augment_deform(page_img)),
            ("torn", lambda page_img: self.augment_torn(page_img)),
            ("dirty", lambda page_img: self.augment_dirty(page_img)),
        ]
        selected_augmentations = random.sample(
            possible_augmentations, random.randint(1, 3)
        )
        print("Selected augmentations:", [name for name, _ in selected_augmentations])
        rotation_angle = 0
        for name, func in selected_augmentations:
            page_img, param = func(page_img)
            if name == "rotate":
                rotation_angle = param
        page_img = self.apply_background(page_img)
        return page_img, rotation_angle

    def save_transcript(self, page_text, base_dir, page_file_name):
        transcript_dir = base_dir / "transcriptions"
        transcript_dir.mkdir(parents=True, exist_ok=True)

        transcript_path = transcript_dir / f"{page_file_name}.txt"
        transcript_path.write_text(page_text, encoding="utf-8")

    def augment_brightness(self, page_img):
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Brightness(page_img)
        page_img = enhancer.enhance(factor)
        return page_img, factor

    def augment_contrast(self, page_img):
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Contrast(page_img)
        page_img = enhancer.enhance(factor)
        return page_img, factor

    def augment_sharpness(self, page_img):
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Sharpness(page_img)
        page_img = enhancer.enhance(factor)
        return page_img, factor

    def augment_rotate(self, page_img):
        angle = random.uniform(-5, 5)
        page_img = page_img.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        return page_img, angle

    def augment_distort(self, page_img):
        page_img_np = np.array(page_img)
        distorted_page_img = distort_line(page_img_np)
        return Image.fromarray(distorted_page_img), None

    def augment_deform(self, page_img: Image.Image):
        page_img_np = np.array(page_img)
        deformed_page_img = deform_image(page_img_np)
        return Image.fromarray(deformed_page_img), None

    def augment_torn(self, page_img):
        page_img = apply_realistic_torn_effect(page_img, num_tears=5, tear_size=30)
        return page_img, None

    def augment_dirty(self, page_img):
        page_img_np = np.array(page_img)
        height, width, _ = page_img_np.shape
        num_spots = random.randint(1, 10)

        for _ in range(num_spots):
            # Randomize the position, size, and shape of the dirty spot
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            size = random.randint(10, 30)  # Size of the dirty spot
            ellipse_width = random.randint(size // 2, size)  # Width of the ellipse
            ellipse_height = random.randint(size // 2, size)  # Height of the ellipse

            # Create an image with the dirty spot
            dirty_spot = Image.new("L", (ellipse_width, ellipse_height), 0)
            draw = ImageDraw.Draw(dirty_spot)
            draw.ellipse((0, 0, ellipse_width, ellipse_height), fill=255)

            # Convert to numpy array
            dirty_spot_np = np.array(dirty_spot)

            # Define region of interest in the original image
            x1 = max(0, x - ellipse_width // 2)
            y1 = max(0, y - ellipse_height // 2)
            x2 = min(width, x + ellipse_width // 2)
            y2 = min(height, y + ellipse_height // 2)

            # Define the region of the dirty spot that fits within the image
            spot_x1 = max(0, ellipse_width // 2 - x)
            spot_y1 = max(0, ellipse_height // 2 - y)
            spot_x2 = spot_x1 + (x2 - x1)
            spot_y2 = spot_y1 + (y2 - y1)

            # Create a mask for the dirty spot
            mask = np.zeros((height, width), dtype=np.uint8)
            if (y2 - y1) > 0 and (x2 - x1) > 0:
                dirty_spot_region = dirty_spot_np[spot_y1:spot_y2, spot_x1:spot_x2]
                mask[y1:y2, x1:x2] = dirty_spot_region

            # Apply the dirty spot to the image
            page_img_np[:, :, 0] = np.where(
                mask > 0, 0, page_img_np[:, :, 0]
            )  # Apply dirty spot to Red channel
            page_img_np[:, :, 1] = np.where(
                mask > 0, 0, page_img_np[:, :, 1]
            )  # Apply dirty spot to Green channel
            page_img_np[:, :, 2] = np.where(
                mask > 0, 0, page_img_np[:, :, 2]
            )  # Apply dirty spot to Blue channel

        return Image.fromarray(page_img_np), None

    def save_image(self, page_img, output_file):
        try:
            page_img.save(output_file, format="JPEG", quality=95)
        except OSError as e:
            print(f"Error saving image {output_file}: {e}")

    def generate_augmented_page_and_line(self):

        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        page_images_dir = self.output_dir / "augmented_page_images"
        line_images_dir = self.output_dir / "line_images"

        page_images_dir.mkdir(parents=True, exist_ok=True)
        line_images_dir.mkdir(parents=True, exist_ok=True)

        self.text_folder = Path(self.text_folder)
        for text_file in self.text_folder.glob("*.txt"):
            page_text = text_file.read_text(encoding="utf-8")
            pages = pgobject.get_pages(page_text)

            text_file_name = text_file.stem  # Get the text file name without extension
            # Extract font name from font path
            font_path = Path(pgobject.font_path)
            font_name = font_path.stem
            for page_num, page_text in enumerate(pages):
                page_img = pgobject.generate_page_image(page_text)
                page_img, rotation_angle = self.apply_augmentation(page_img)
                page_img = page_img.convert("RGB")
                page_file_name = f"{text_file_name}_{font_name}_page_{page_num}"
                augmented_page_file = page_images_dir / f"{page_file_name}.jpg"
                angle_suffix = (
                    f"_angle_{int(rotation_angle)}" if rotation_angle != 0 else ""
                )
                output_file_with_angle = augmented_page_file.with_name(
                    augmented_page_file.stem + angle_suffix + augmented_page_file.suffix
                )
                page_img.save(output_file_with_angle, format="JPEG", quality=95)

                # Extract and save line images with rotation angle suffix
                lines = pgobject.extract_lines(page_img, page_text, rotation_angle)
                for j, line_page_img in enumerate(lines):
                    line_page_img_file = (
                        line_images_dir
                        / f"{page_file_name}_line_{j}_angle_{int(rotation_angle)}.jpg"
                    )
                    self.save_image(line_page_img, line_page_img_file)

                # Save the transcript
                self.save_transcript(page_text, self.output_dir, page_file_name)


if __name__ == "__main__":
    generated_augmented_output_dir = Path(__file__).parent / "../.." / "data" / "output"
    generated_augmented_output_dir.mkdir(parents=True, exist_ok=True)
    generated_augmented_path = str(generated_augmented_output_dir)
    text_folder = Path(__file__).parent / "../.." / "data" / "texts" / "kangyur"
    generator = PageAugmentationGenerator(text_folder, generated_augmented_path)
    generator.generate_augmented_page_and_line()
