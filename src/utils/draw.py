from ..dataloader import Dataloader
from PIL import Image, ImageDraw
import multiprocessing
import progressbar
import threading
import time
import os


def draw_labels(
    datasets: list[str],
    lang: list[str]
) -> None:
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
            lang = lang
        )
    print("Dataloader created\n")

    print("Draw labels")
    for dataset in dataloaders.keys():
        draw_folder_path = os.path.join("draw", dataset)
        os.makedirs(draw_folder_path, exist_ok=True)

        args = []
        dataloader = dataloaders[dataset]
        for split in ["train", "test"]:
            split_draw_path = os.path.join(draw_folder_path, split)
            os.makedirs(split_draw_path, exist_ok=True)
            for (img_path, labels) in dataloader.data[split]:
                args.append((
                    img_path,
                    labels,
                    split_draw_path,
                    "black",
                    1
                ))

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
    color: str,
    width: int
) -> None:
    _, img_name = os.path.split(img_path)
    img = Image.open(img_path)
    if img.mode == 'RGBA':
        img = img.convert('RGB')

    draw = ImageDraw.Draw(img)

    for (_, bbox) in labels:
        draw.rectangle(
            xy = bbox,
            outline = color,
            width = width
        )

    img.save(os.path.join(output_dir, img_name))
    img.close()