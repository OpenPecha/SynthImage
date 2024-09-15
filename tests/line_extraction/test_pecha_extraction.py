from pathlib import Path

import cv2
import numpy as np

from SynthImage.LineExtraction.line_extraction import extract_lines_from_image


def test_extract_lines_from_image():
    # Load the test image
    image_path = Path(
        "tests/data/pecha_format_page_img/794x265_pecha_format_page_image_without_bbox.png"
    )
    image = cv2.imread(str(image_path))

    # Extract lines from the image
    filename = image_path.name
    page_number = 1
    extracted_lines = extract_lines_from_image(image, filename, page_number)

    # Assertions to verify the extracted lines
    assert len(extracted_lines) > 0, "No lines were extracted from the image"

    for i, line_img in enumerate(extracted_lines):
        # Check if each extracted line is a valid image
        assert isinstance(
            line_img, np.ndarray
        ), f"Extracted line {i} is not a numpy array"
        assert (
            line_img.shape[0] > 0 and line_img.shape[1] > 0
        ), f"Extracted line {i} has invalid dimensions"

        # Save each extracted line for visual inspection
        output_dir = Path("./tests/output/extracted_lines")
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / f"line_{i}.png"), line_img)

    # Check if the number of extracted lines is reasonable (you may need to adjust this based on your specific image)
    assert (
        5 <= len(extracted_lines) <= 20
    ), f"Unexpected number of lines extracted: {len(extracted_lines)}"

    # You can add more specific assertions here based on your knowledge of the test image
    # For example, checking the dimensions of specific lines if you know what they should be

    print(
        f"Successfully extracted and verified {len(extracted_lines)} lines from the image."
    )
