from collections import defaultdict
from pathlib import Path
from typing import DefaultDict

from SynthImage.SynthPageImage.modern_book_format_page_image import (
    ModernBookPageGenerator,
)

if __name__ == "__main__":
    # Path to the text file
    text_file_path = "./data/texts/kangyur/v001_plain.txt"

    # Path to the font folder containing multiple font files
    font_folder_path = Path("./data/fonts/Drutsa_short/")
    font_sizes = [12, 14, 16, 30]  # Adjust the font size as needed
    dimensions = [
        (626, 771),
        (1063, 1536),
        (259, 194),
        (349, 522),
        (974, 1500),
        (968, 1440),
    ]
    # Define probabilities for each dimension
    dimension_probs = [0.3, 0.3, 0.1, 0.1, 0.1, 0.1]

    # Define probabilities for each font size
    font_size_probs = {12: 0.5, 14: 0.2, 16: 0.15, 30: 0.15}
    # Read the volume text from the file
    with open(text_file_path, encoding="utf-8") as file:
        vol_text = file.read()

    # Initialize the PageGenerator
    modern_book_page_generator = ModernBookPageGenerator(
        left_padding=50,
        right_padding=50,
        top_padding=50,
        bottom_padding=50,
        dimensions=dimensions,
    )

    # Iterate over all font files in the font folder
    for font_path in font_folder_path.glob("*.ttf"):
        font_name = font_path.stem  # Extract the font name from the font path

        # Generate page images for the current font
        pages, dimension_count = modern_book_page_generator.generate_modern_page_images(
            vol_text, font_sizes, font_path, dimension_probs, font_size_probs
        )

        # Define the output path for the current font
        output_path = Path(f"./data/modern_page_output/{font_name}")
        output_path.mkdir(parents=True, exist_ok=True)

        # Track the count of images for each dimension
        dimension_counter: DefaultDict[str, int] = defaultdict(int)
        for i, (page_img, font_size) in enumerate(pages):
            # Get the dimension of the current page image
            page_width, page_height = page_img.size

            # Create the filename with dimension, font size, and font name
            dimension_prefix = f"{page_width}x{page_height}"
            # Increment the dimension counter
            dimension_counter[dimension_prefix] += 1
            count = dimension_counter[dimension_prefix]

            filename = f"page_{i+1}_{dimension_prefix}_count_{count}_font{font_size}_{font_name}.png"

            # Save the image
            page_img.save(output_path / filename)

        # Print the count of pages for each dimension
        for dimension, count in dimension_count.items():
            print(f"Font: {font_name} - Dimension {dimension}: {count} pages generated")
