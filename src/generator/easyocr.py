from .generator import Generator
from ..dataloader import Dataloader
from ..utils.image import open_image
import pandas as pd
import cv2
import os


class EasyOCRGenerator(Generator):
    def __init__(
        self,
        test_name: str,
        datasets: list, 
        dict: list[str] | None,
        workers: int,
        augmentation: bool
    ) -> None:
        super().__init__(
            test_name,
            datasets,
            dict,
            workers,
            augmentation
        )

    def _generate(
        self,
        dataloader: Dataloader,
        task: str
    ) -> None:
        root_path = os.path.join(self.root_path, task)
        os.makedirs(root_path, exist_ok=True)

        for split in ["train", "test"]:
            if task == "Detection":
                img_output_path = os.path.join(root_path, f"{split}ing_images")
                label_output_path = os.path.join(root_path, f"{split}ing_localization_transcription_gt")
                os.makedirs(img_output_path, exist_ok=True)
                os.makedirs(label_output_path, exist_ok=True)

                self.run_process(img_output_path, dataloader, task, split)

    def _det(
        self,
        img_output_path: str,
        img_path: str,
        gt: list,
        transform: tuple[str, callable] = (None, None)
    ) -> None:
        img = open_image(img_path, transform[1])
        _, img_name = os.path.split(img_path)
        img_id, _ = tuple(img_name.split("."))

        label_output_path = img_output_path.replace("images", "localization_transcription_gt")

        if transform[0] is not None:
            img_name = img_name.replace(".", f"-{transform[0]}.")

        with open(os.path.join(label_output_path, f"{img_id}.txt"), "w") as file:
            for (text, bbox) in gt:
                x1, y1, x2, y2 = tuple(bbox)
                file.write(f"{x1},{y1},{x2},{y1},{x2},{y2},{x1},{y2},{text}\n")

        cv2.imwrite(os.path.join(img_output_path, img_name), img)

    def _rec(
        self,
        img_output_path: str,
        img_path: str,
        gt: list,
        transform: tuple[str, callable] = (None, None)
    ) -> None:
        img = open_image(img_path, transform[1])
        _, img_name = os.path.split(img_path)

        result = []
        for i, (text, bbox) in enumerate(gt):
            crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if transform[0] is not None:
                crop_name = img_name.replace(".", f"-{i}-{transform[0]}.")
            else:
                crop_name = img_name.replace(".", f"-{i}.")   
            cv2.imwrite(os.path.join(img_output_path, crop_name),crop)

            result.append((crop_name, text))

        return result