from PIL import Image
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
        cv_image (numpy.ndarray): Image in OpenCV format.
    """
    img = Image.open(img_path)
    if img.mode in ["RGBA", "LA"]:
        img = img.convert("RGB")

    cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img.close()

    if transform is not None:
        cv_image = transform(image=cv_image)["image"]
        
    return cv_image