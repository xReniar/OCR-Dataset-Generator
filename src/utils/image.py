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

    try:
        deg = {3:180,6:270,8:90}.get(img.getexif().get(274,0),0)
    except:
        deg = 0

    if deg != 0:
        img = img.rotate(deg, expand=False)
    cv_image = np.array(img)
    if len(cv_image.shape)==3 and cv_image.shape[2] >= 3:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) 
    img.close()
    
    if transform is not None:
        cv_image = transform(image=cv_image)["image"]
    
    return cv_image