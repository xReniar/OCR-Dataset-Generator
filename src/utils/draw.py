from ..dataloader import Dataloader
from .image import open_image
from yaspin import yaspin
from yaspin.spinners import Spinners
from colorama import Fore
import multiprocessing
import time
import cv2
import os


def draw_labels(
    datasets: list[str],
    lang: list[str],
    workers: int,
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
        
    print(f"{Fore.LIGHTCYAN_EX}[Dataloader creation]{Fore.RESET}")
    with yaspin(text=f"Creating dataloader", spinner=Spinners.line) as spinner:
        dataloaders = {}
        for dataset in new_datasets:
            dataloaders[dataset] = Dataloader(
                datasets = [dataset],
                dict = lang
            )
        spinner.text = "Dataloader created\n"
        spinner.ok(f"{Fore.GREEN}✓{Fore.RESET}")

    print(f"{Fore.LIGHTCYAN_EX}[Label drawing]{Fore.RESET}")
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

        with yaspin(text=f"Drawing labels in \"draw/{dataset}\"", spinner=Spinners.line) as spinner:
            start_time = time.time()
            with multiprocessing.Pool(processes=workers) as pool:
                pool.starmap(draw_single_img, args)
            end_time = time.time()

            spinner.text = f"Finished drawing \"{dataset}\" labels in {end_time - start_time:.2f}s"
            spinner.ok(f"{Fore.GREEN}✓{Fore.RESET}")

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