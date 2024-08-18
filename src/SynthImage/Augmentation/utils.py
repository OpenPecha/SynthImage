import os

from PIL import Image, ImageDraw, ImageFont

# Configuration
PAGE_WIDTH = 800
PAGE_HEIGHT = 1200
LEFT_PADDING = 50
RIGHT_PADDING = 50
TOP_PADDING = 50
BOTTOM_PADDING = 50
FONT_PATH = "/path/to/font.ttf"  # Update this path
FONT_SIZE = 24
LINE_SPACING = 10
PARAGRAPH_SPACING = 20
OUTPUT_DIR = "output_pages"


def create_blank_page():
    """Create a blank page image with specified dimensions and background color."""
    return Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")


def render_text_on_page(text, output_dir, page_number):
    """Render text on the blank page and save the image."""
    # Create a blank page image
    page_image = create_blank_page()
    draw = ImageDraw.Draw(page_image)

    # Load the font
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    y_position = TOP_PADDING

    # Calculate available width for text
    available_width = PAGE_WIDTH - LEFT_PADDING - RIGHT_PADDING

    # Split text into lines
    lines = text.splitlines()
    for line in lines:
        words = line.split()
        current_line = ""
        for word in words:
            # Measure the width of the current line with the new word
            test_line = current_line + word + " "
            line_width = draw.textbbox((0, 0), test_line, font=font)[2]

            # Calculate the remaining width
            current_line_width = draw.textbbox((0, 0), current_line, font=font)[2]
            remaining_width = available_width - current_line_width

            if line_width > available_width or remaining_width < LEFT_PADDING:
                # Draw the current line on the page
                draw.text(
                    (LEFT_PADDING, y_position),
                    current_line.strip(),
                    font=font,
                    fill="black",
                )
                y_position += FONT_SIZE + LINE_SPACING

                # Check if adding this line exceeds the page height
                if y_position + FONT_SIZE > PAGE_HEIGHT - BOTTOM_PADDING:
                    # Save the image and start a new page
                    output_image_path = os.path.join(
                        output_dir, f"synthetic_page_{page_number}.png"
                    )
                    page_image.save(output_image_path)
                    page_number += 1
                    page_image = create_blank_page()
                    draw = ImageDraw.Draw(page_image)
                    y_position = TOP_PADDING

                # Start a new line with the current word
                current_line = word + " "
            else:
                # Continue adding to the current line
                current_line = test_line

        # Draw the last line if it has content
        if current_line.strip():
            draw.text(
                (LEFT_PADDING, y_position),
                current_line.strip(),
                font=font,
                fill="black",
            )
            y_position += FONT_SIZE + LINE_SPACING

        # Add space after the paragraph
        y_position += PARAGRAPH_SPACING

        # Start a new page if the next line won't fit
        if y_position + FONT_SIZE > PAGE_HEIGHT - BOTTOM_PADDING:
            output_image_path = os.path.join(
                output_dir, f"synthetic_page_{page_number}.png"
            )
            page_image.save(output_image_path)
            page_number += 1
            page_image = create_blank_page()
            draw = ImageDraw.Draw(page_image)
            y_position = TOP_PADDING

    # Save the last page if it has content
    if y_position > TOP_PADDING:
        output_image_path = os.path.join(
            output_dir, f"synthetic_page_{page_number}.png"
        )
        page_image.save(output_image_path)


def generate_pages_from_text(text_file_path, output_dir):
    """Generate pages from a text file and save them."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the text from the file
    with open(text_file_path, encoding="utf-8") as file:
        text = file.read()

    # Render text onto pages
    render_text_on_page(text, output_dir, 1)


# Example usage
generate_pages_from_text("/path/to/text_file.txt", OUTPUT_DIR)
