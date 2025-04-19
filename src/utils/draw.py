from ..dataloader import Dataloader
from .image import open_image
import multiprocessing
import progressbar
import threading
import time
import cv2
import os


def draw_labels(
    datasets: list[str],
    lang: list[str],
    color: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 1
) -> None:
    """
    Draws the labels of the images in the specified datasets.
    
    Args:
        datasets (list[str]): List of dataset names.
        lang (list[str]): List of languages.
    """
    new_datasets = []
    for dataset in datasets:
        if "-" in dataset:
            root = dataset.split("-")[0]
            new_datasets.append(os.path.join(root, dataset))
        else:
            new_datasets.append(dataset)
    
    print("\nCreating dataloader")
    dataloaders = {}
    for dataset in new_datasets:
        dataloaders[dataset] = Dataloader(
            datasets = [dataset],
            dict = lang
        )
    print("Dataloader created\n")

    print("Draw labels")
    for dataset in dataloaders.keys():
        draw_folder_path = os.path.join("draw", dataset)
        os.makedirs(draw_folder_path, exist_ok=True)

        args = []
        dataloader: Dataloader = dataloaders[dataset]
        for split in ["train", "test"]:
            split_draw_path = os.path.join(draw_folder_path, split)
            os.makedirs(split_draw_path, exist_ok=True)
            for (img_path, labels) in dataloader.data[split]:
                args.append((img_path, labels, split_draw_path, color, thickness))

        drawing = True
        def progress_bar():
            widgets = ["  [", progressbar.AnimatedMarker(), f"] Drawing labels in \"draw/{dataset}\""]
            bar = progressbar.ProgressBar(widgets=widgets, maxval=progressbar.UnknownLength).start()
            i = 0
            while drawing:
                i += 1
                bar.update(i)
                time.sleep(0.1)
            bar.widgets =  [f"  [✓] Finished drawing \"{dataset}\" labels"]
            bar.finish()

        progress_thread = threading.Thread(target=progress_bar)
        progress_thread.start()

        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            pool.starmap(draw_single_img, args)

        drawing = False
        progress_thread.join()

def draw_single_img(
    img_path: str,
    labels: list,
    output_dir: str,
    color: tuple[int, int, int],
    thickness: int
) -> None:
    """
    Draws labels on an image and saves it to the output directory.

    Args:
        img_path (str): Path to the input image.
        labels (list): List of labels to draw on the image.
        output_dir (str): Directory to save the output image.
        color (tuple[int, int, int]): Color of the bounding boxes in BGR format.
        thickness (int): Thickness of the bounding boxes.
    """
    _, img_name = os.path.split(img_path)
    img = open_image(img_path)

    for (_, bbox) in labels:
        cv2.rectangle(
            img,
            pt1 = (int(bbox[0]), int(bbox[1])),
            pt2 = (int(bbox[2]), int(bbox[3])),
            color = color,
            thickness = thickness
        )

    cv2.imwrite(os.path.join(output_dir, img_name), img)