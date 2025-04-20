from PIL import Image, ImageOps
import numpy as np
import cv2


def open_image(
    img_path: str,
    transform: callable = None
) -> np.ndarray:
    """
    Open an image file and convert it to a format suitable for processing.

    Args:
        img_path (str): Path to the image file.

    Returns:
        cv2_img (numpy.ndarray): Image in OpenCV format.
    """
    img = ImageOps.exif_transpose(Image.open(img_path))
    if img.mode in ["RGBA", "LA"]:
        img = img.convert("RGB")

    cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img.close()

    cv2_img = cv2.imread(img_path)
    
    if transform is not None:
        cv2_img = transform(image=cv2_img)["image"]
    
    return cv2_img