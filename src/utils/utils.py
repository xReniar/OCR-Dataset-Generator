import os
import ast


def read_label(
    label_path:str
) -> list[tuple[str, list[int]]]:
    label_content = []
    with open(label_path, "r") as label_file:
        rows = list(map(lambda row: row.strip("\n").split("\t"), label_file.readlines()))
        for (text, bbox) in rows:
            label_content.append((text, ast.literal_eval(bbox)))
    
    return label_content


def read_labels(label_dir_path:str) -> dict[str, list[tuple[str, list[int]]]]:
    labels_content = {}
    for label_filename in os.listdir(label_dir_path):
        labels_content[label_filename] = read_label(os.path.join(label_dir_path, label_filename))

    return labels_content