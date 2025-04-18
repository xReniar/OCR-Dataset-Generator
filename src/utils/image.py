from PIL import Image
import numpy as np
import cv2


def open_image(
    img_path: str
) -> Image.Image:
    """
    Open an image file and convert it to a format suitable for processing.

    Args:
        img_path (str): Path to the image file.

    Returns:
        Image.Image: The opened and converted image.
    """
    img = Image.open(img_path)
    if img.mode in ["RGBA", "LA"]:
        img = img.convert("RGB")
        
    return img