from .reader import read_labels
import os


def check_images(
    dataset_path:str,
) -> dict:
    errors = {}
    for split in ["train", "test"]:
        images_dir = set(map(lambda fn: fn.split(".")[0], sorted(os.listdir(os.path.join(dataset_path, split, "images")))))
        labels_dir = set(map(lambda fn: fn.split(".")[0], sorted(os.listdir(os.path.join(dataset_path, split, "labels")))))

        missing_labels = images_dir - labels_dir
        if len(missing_labels) < len(images_dir):
            errors["missing_labels"] = list(missing_labels)
        else:
            errors["missing_labels"] = []

        missing_images = labels_dir - images_dir
        if len(missing_images) < len(labels_dir):
            errors["missing_images"] = list(missing_images)
        else:
            errors["missing_images"] = []
    
    return errors

def check_labels(
    dataset_path: str,
) -> None:
    errors = {}
    for split in ["train", "test"]:
        label_dir_path = os.path.join(dataset_path, split, "labels")

        labels_content = read_labels(label_dir_path)
        for label_filename in labels_content.keys():
            for i, (text, bbox) in enumerate(labels_content[label_filename]):
                x1, y1, x2, y2 = bbox
                if not(x1 < x2 and y1 < y2):
                    errors[os.path.join(dataset_path, split, "labels", label_filename)] = dict(
                        line = i + 1,
                        text = text,
                        bbox = [x1, y1, x2, y2]
                    )
    return errors