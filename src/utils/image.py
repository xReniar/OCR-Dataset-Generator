from PIL import Image


def open_image(
    img_path: str,
    mode: str = "RGB"
) -> Image.ImageFile:
    """
    Open an image and convert it to the specified mode.

    Args:
        img_path (str): Path to the image file.
        mode (str): Mode to convert the image to. Default is "RGB".

    Returns:
        Image.Image: The opened and converted image.
    """
    img = Image.open(img_path)
    if img.mode in ['RGBA', 'LA']:
        img = img.convert(mode)
        
    return img