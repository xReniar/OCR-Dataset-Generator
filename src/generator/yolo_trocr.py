from .generator import Generator
from ..dataloader import Dataloader
from PIL import Image
import multiprocessing
import os


class YOLOTrOCRGenerator(Generator):
    def __init__(
        self,
        test_name: str,
        datasets: list,
        lang: list[str] | None,
        workers: int,
        transforms = None
    ) -> None:
        super().__init__(
            test_name,
            datasets,
            lang,
            workers,
            transforms
        )

    def _generate(
        self,
        dataloader: Dataloader,
        task: str,
        process
    ) -> None:
        root_path = os.path.join(self.root_path, task)
        os.makedirs(root_path, exist_ok=True)

        for split in ["train", "test"]:
            img_output_path = os.path.join(root_path, split)
            os.makedirs(img_output_path, exist_ok=True)
            if task == "Detection":
                os.makedirs(os.path.join(img_output_path, "images"), exist_ok=True)
                os.makedirs(os.path.join(img_output_path, "labels"), exist_ok=True)

            args = [(img_output_path, img_path, gt) for (img_path, gt) in dataloader.data[split]]

            with multiprocessing.Pool(processes=self.workers) as pool:
                results = pool.starmap(process, args)

            if task == "Recognition":
                with open(os.path.join(root_path, f"{split}.txt"), "w") as file:
                    results = [item for sublist in results for item in sublist]
                    file.writelines(results)

    def _det(
        self,
        img_output_path: str,
        img_path: str,
        gt: list
    ) -> None:
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        _, img_name = os.path.split(img_path)
        img_id = img_name.split(".")[0]
        img.save(os.path.join(img_output_path, "images", img_name))
        width, height = img.width, img.height

        with open(os.path.join(img_output_path, "labels", f"{img_id}.txt"), "w") as file:
            for (_, bbox) in gt:
                x1, y1, x2, y2 = bbox
                x_center = (x1 + x2 / 2) / width
                y_center = (y1 + y2 / 2) / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height

                # every text has label 0 (for now)
                # "label_id = func(text)"" or the text is the label itself
                file.write(f"0 {x_center} {y_center} {box_width} {box_height}\n")
        img.close()

    def _rec(
        self,
        img_output_path: str,
        img_path: str,
        gt: list
    ) -> list[str]:
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        _, img_name = os.path.split(img_path)
        result = []
        for i, (text, bbox) in enumerate(gt):
            crop = img.crop(bbox)
            crop_name = img_name.replace(".", f"-{i}.")
            crop.save(os.path.join(img_output_path, crop_name))
            crop.close()

            result.append(f"{crop_name} {text}\n")
        
        return result