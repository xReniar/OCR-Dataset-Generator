from .generator import Generator
from ..dataloader import Dataloader
from ..utils.image import open_image
from pathlib import Path
import json
import os
import cv2


class PaddleOCRGenerator(Generator):
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
            img_output_path = os.path.join(root_path, split)
            os.makedirs(img_output_path, exist_ok=True)

            results = self.run_process(img_output_path, dataloader, task, split)

            with open(os.path.join(root_path, f"{split}_label.txt"), "w") as file:
                if task == "Recognition":
                    results = [item for sublist in results for item in sublist]
                    results = [f"{path}\t{text}\n" for (path, text) in results]
                file.writelines(results)
                
    def _det(
        self,
        img_output_path: str,
        img_path: str,
        gt: list,
        transform: tuple[str, callable] = (None, None)
    ) -> str:
        img = open_image(img_path, transform[1])
        _, img_name = os.path.split(img_path)

        if transform[0] is not None:
            img_name = img_name.replace(".", f"-{transform[0]}.")

        cv2.imwrite(os.path.join(img_output_path, img_name), img)

        split = Path(img_output_path).parts[-1]
        annotations = []
        for (text, bbox) in gt:
            x1, y1, x2, y2 = bbox
            annotations.append(dict(
                transcription=text,
                points = [[x1, y1],[x2, y1],[x2, y2],[x1, y2]]
            ))

        return f"{os.path.join('Detection', split, img_name)}\t{json.dumps(annotations, ensure_ascii=False)}\n"

    def _rec(
        self,
        img_output_path: str,
        img_path: str,
        gt: list,
        transform: tuple[str, callable] = (None, None)
    ) -> list[tuple[str, str]]:
        img = open_image(img_path, transform[1])
        _, img_name = os.path.split(img_path)

        split = Path(img_output_path).parts[-1]
        result = []
        for i, (text, bbox) in enumerate(gt):
            crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if transform[0] is not None:
                crop_name = img_name.replace(".", f"-{i}-{transform[0]}.")
            else:
                crop_name = img_name.replace(".", f"-{i}.")
                
            cv2.imwrite(os.path.join(img_output_path, crop_name), crop)
            result.append((os.path.join("Recognition", split, crop_name), text))
        
        return result