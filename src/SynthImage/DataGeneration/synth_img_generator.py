import os
import random

from SynthImage.SynthPageImg.page_image import PageGenerator


def main():
    text_file_path = "./data/texts/kangyur/v001_plain.txt"
    fonts_folder = "/Users/ogyenthoga/Desktop/Work/SynthImage/data/fonts"
    output_dir = "/Users/ogyenthoga/Desktop/Work/SynthImage/data/SynthPageImages"

    # Extract the base name of the text file (without extension) to use as a prefix
    text_file_name = os.path.basename(text_file_path).split(".")[0]

    # Read the content of the text file
    with open(text_file_path, encoding="utf-8") as file:
        vol_text = file.read()

    # Font sizes to randomly choose from
    font_sizes = [20, 25, 30, 35]

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the PageGenerator (shared across different font sizes)
    pgobject = PageGenerator(
        left_padding=80,
        right_padding=80,
        top_padding=40,
        bottom_padding=40,
    )

    # Iterate through each font file in the fonts folder
    for subfolder in os.listdir(fonts_folder):
        subfolder_path = os.path.join(fonts_folder, subfolder)

        if os.path.isdir(subfolder_path):
            for font_file in os.listdir(subfolder_path):
                if font_file.endswith(".ttf") or font_file.endswith(".otf"):
                    font_path = os.path.join(subfolder_path, font_file)

                    # Generate images with a different random font size for each page
                    pages = pgobject.get_pages(vol_text)
                    for i, page_text in enumerate(pages):
                        # Randomize font size for each page
                        font_size = random.choice(font_sizes)

                        # Generate the page image
                        page_image = pgobject.generate_page_image(
                            page_text, font_path, font_size
                        )

                        # Save the generated image with the font size and page sequence in the file name
                        font_name = os.path.basename(font_path).split(".")[0]
                        output_filename = f"{text_file_name}_size{font_size}_{subfolder}_{font_name}_synthetic_page_{i+1:04d}.png"  # noqa
                        output_path = os.path.join(output_dir, output_filename)

                        page_image.save(output_path)  # Save the image
                        print(
                            f"Saved image {output_filename} to {output_path}."
                        )  # Debug: Print save info


if __name__ == "__main__":
    main()
