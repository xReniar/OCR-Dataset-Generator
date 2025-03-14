from .reader import read_labels
import multiprocessing
import os
import cv2


def draw_labels(
    dataset_path: str
) -> None:
    os.makedirs(os.path.join(dataset_path,"draw"), exist_ok=True)
    all_tasks = []

    # create tasks
    for split in ["train", "test"]:
        label_dir = os.path.join(dataset_path, split, "labels")
        img_dir = os.path.join(dataset_path, split, "images")
        output_dir = os.path.join(dataset_path, "draw", split)
        os.makedirs(output_dir, exist_ok=True)
        
        labels_content = read_labels(label_dir)
        for label_filename in labels_content.keys():
            all_tasks.append((
                label_filename,
                labels_content[label_filename],
                img_dir,
                output_dir,
                (0, 0, 0),
                1
            ))

    # start multiprocessing
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        pool.starmap(draw_single_img, all_tasks)

def draw_single_img(
    label_filename: str,
    label_data: str,
    img_dir: str,
    output_dir: str,
    color: tuple,
    thickness: int
) -> None:
    img_name = label_filename.strip(".txt")
    
    img = cv2.imread(os.path.join(img_dir, img_name))

    for (_, bbox) in label_data:
        img = cv2.rectangle(
            img = img,
            pt1 = (bbox[0], bbox[1]),
            pt2 = (bbox[2], bbox[3]),
            color = color,
            thickness = thickness
        )

    cv2.imwrite(os.path.join(output_dir, img_name), img)