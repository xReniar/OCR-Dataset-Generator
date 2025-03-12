import os
import ast


def read_label(
    label_path:str
) -> tuple[str, list[tuple[str, list[int]]]]:
    label_content = []
    with open(label_path, "r") as label_file:
        rows = list(map(lambda row: row.strip("\n").split("\t"), label_file.readlines()))
        for (text, bbox) in rows:
            label_content.append((text, ast.literal_eval(bbox)))
    
    return (label_path.strip(".txt").replace("labels", "images"), label_content)

def read_labels(
    label_dir_path:str
) -> dict[str, list[tuple[str, list[int]]]]:
    labels_content = {}
    for label_filename in os.listdir(label_dir_path):
        _, label_content = read_label(os.path.join(label_dir_path, label_filename))
        labels_content[label_filename] = label_content

    return labels_content