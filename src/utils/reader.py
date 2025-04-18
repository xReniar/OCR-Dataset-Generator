import os
import ast


def read_label(
    label_path:str
) -> tuple[str, list[tuple[str, list[int]]]]:
    """
    Reads the label file and returns the image path and the labels.

    Args:
        label_path (str): Path to the label file.

    Returns:
        tuple[str, list[tuple[str, list[int]]]]: A tuple containing the image path and a list of tuples with text and bounding box coordinates.
    """
    label_content = []
    with open(label_path, "r") as label_file:
        rows = list(map(lambda row: row.strip("\n").split("\t"), label_file.readlines()))
        for (text, bbox) in rows:
            label_content.append((text, ast.literal_eval(bbox)))
    
    return (label_path.strip(".txt").replace("labels", "images"), label_content)

def read_labels(
    label_dir_path:str
) -> dict[str, list[tuple[str, list[int]]]]:
    """
    Reads all label files in a directory and returns a dictionary with image paths as keys and label contents as values.

    Args:
        label_dir_path (str): Path to the directory containing label files.

    Returns:
        dict[str, list[tuple[str, list[int]]]]: A dictionary with image paths as keys and label contents as values.
    """
    labels_content = {}
    for label_filename in os.listdir(label_dir_path):
        _, label_content = read_label(os.path.join(label_dir_path, label_filename))
        labels_content[label_filename] = label_content

    return labels_content