# generate_pages.py

from pathlib import Path

from SynthImage.SynthPageImage.pecha_format_page_image import PageGenerator

if __name__ == "__main__":
    text_file_path = (
        "/Users/ogyenthoga/Desktop/Work/SynthImage/data/texts/kangyur/v001_plain.txt"
    )
    font_path = "/Users/ogyenthoga/Desktop/Work/SynthImage/data/fonts/Drutsa short/Kangba Derchi-Drutsa.ttf"
    font_size = 12  # Adjust the font size as needed
    chars_per_line = 260
    page_width = 1123
    page_height = 265

    # Read the volume text from the file
    with open(text_file_path, encoding="utf-8") as file:
        vol_text = file.read()

    # Initialize the PageGenerator
    page_generator = PageGenerator(
        left_padding=50,
        right_padding=50,
        top_padding=50,
        bottom_padding=50,
        chars_per_line=chars_per_line,
    )

    # Generate page images from the volume text
    pages = page_generator.generate_page_images(
        vol_text, font_size, font_path, page_width, page_height
    )

    # Save each page image to the output directory
    output_path = Path("/Users/ogyenthoga/Desktop/Work/SynthImage/data/output")
    output_path.mkdir(parents=True, exist_ok=True)

    for i, page_image in enumerate(pages):
        page_image_path = output_path / f"output_page_{i + 1}.png"
        page_image.save(page_image_path)
