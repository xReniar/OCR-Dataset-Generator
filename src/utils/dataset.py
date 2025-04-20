from .reader import read_labels
import os


def check_images(
    dataset_path:str,
) -> dict:
    """
    Check if the images and labels are correct. The images and labels are correct if:
    - The images have a corresponding label.
    - The labels have a corresponding image.

    Args:
        dataset_path (str): Path to the dataset.
        
    Returns:
        dict: Dictionary containing the errors. The keys are the paths to the images and labels and the values are
        dictionaries containing the missing images and labels.
    """
    errors = {}
    missing_labels = []
    missing_images = []
    for split in ["train", "test"]:
        images_dir = set(map(lambda fn: fn.split(".")[0], sorted(os.listdir(os.path.join(dataset_path, split, "images")))))
        labels_dir = set(map(lambda fn: fn.split(".")[0], sorted(os.listdir(os.path.join(dataset_path, split, "labels")))))

        missing_labels += list(map(lambda fn: f"{split}/labels/{fn}", list(images_dir - labels_dir)))
        missing_images += list(map(lambda fn: f"{split}/images/{fn}", list(labels_dir - images_dir)))

    errors["missing_labels"] = missing_labels
    errors["missing_images"] = missing_images
    
    return errors

def check_labels(
    dataset_path: str,
) -> dict:
    """
    Check if the labels are correct. The labels are correct if:
    - The bounding box coordinates are correct (x1 < x2 and y1 < y2).
    - The text is not empty.

    Args:
        dataset_path (str): Path to the dataset.

    Returns:
        dict: Dictionary containing the errors. The keys are the paths to the labels and the values are dictionaries
        containing the line number, text and bounding box coordinates.
    """
    errors = {}
    for split in ["train", "test"]:
        label_dir_path = os.path.join(dataset_path, split, "labels")

        labels_content = read_labels(label_dir_path)
        for label_filename in labels_content.keys():
            for i, (text, bbox) in enumerate(labels_content[label_filename]):
                x1, y1, x2, y2 = bbox
                if not(x1 < x2 and y1 < y2) or any(point < 0 for point in bbox) or len(text) == 0:
                    errors[os.path.join(dataset_path, split, "labels", label_filename)] = dict(
                        line = i + 1,
                        text = text,
                        bbox = [x1, y1, x2, y2]
                    )
    return errors