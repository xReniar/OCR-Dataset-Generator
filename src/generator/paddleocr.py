from .generator import Generator
from ..dataloader import Dataloader
from PIL import Image
from pathlib import Path
import multiprocessing
import os
import json


class PaddleOCRGenerator(Generator):
    def __init__(
        self,
        test_name: str,
        datasets: list,
        lang: list[str] | None,
        transforms = None
    ) -> None:
        super().__init__(
            test_name,
            datasets,
            lang,
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

            args = [(img_output_path, img_path, gt) for (img_path, gt) in dataloader.data[split]]

            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                results = pool.starmap(process, args)

            with open(os.path.join(root_path, f"{split}_label.txt"), "w") as file:
                if task == "Recognition":
                    results = [item for sublist in results for item in sublist]
                    results = [f"{path}\t{text}\n" for (path, text) in results]
                file.writelines(results)
                
    def _det(
        self,
        img_output_path: str,
        img_path: str,
        gt: list
    ) -> str:
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
            
        _, img_name = os.path.split(img_path)
        img.save(os.path.join(img_output_path, img_name))

        split = Path(img_output_path).parts[-1]
        annotations = []
        for (text, bbox) in gt:
            x1, y1, x2, y2 = bbox
            annotations.append(dict(
                transcription=text,
                points = [[x1, y1],[x2, y1],[x2, y2],[x1, y2]]
            ))
        img.close()

        return f"{os.path.join('Detection', split, img_name)}\t{json.dumps(annotations, ensure_ascii=False)}\n"

    def _rec(
        self,
        img_output_path: str,
        img_path: str,
        gt: list
    ) -> list[tuple[str, str]]:
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        _, img_name = os.path.split(img_path)
        split = Path(img_output_path).parts[-1]
        result = []
        for i, (text, bbox) in enumerate(gt):
            crop = img.crop(bbox)
            crop_name = img_name.replace(".", f"-{i}.")
            crop.save(os.path.join(img_output_path, crop_name))
            crop.close()

            result.append((os.path.join("Recognition", split, crop_name), text))
        img.close()
        
        return result