from .generator import Generator
from ..dataloader import Dataloader
import multiprocessing
import cv2
import hashlib
import json
import os


class DoctrGenerator(Generator):
    def __init__(
        self,
        test_name: str,
        datasets: list,
        transforms = None
    ) -> None:
        super().__init__(
            test_name,
            datasets,
            transforms
        )

    def _generate(
        self,
        dataloader: Dataloader,
        task: str,
        process,
    ) -> None:
        root_path = os.path.join(self.root_path, task)
        os.makedirs(root_path, exist_ok=True)

        for split in ["train", "test"]:
            img_output_path = os.path.join(root_path, split, "images")
            os.makedirs(img_output_path, exist_ok=True)

            args = [(img_output_path, img_path, gt) for (img_path, gt) in dataloader.data[split]]

            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                results = pool.starmap(process, args)

            with open(os.path.join(root_path, split, "labels.json"),"w") as file:
                labels = {}
                if task == "Recognition":
                    results = [item for sublist in results for item in sublist]
                for (img_name, label) in results:
                    labels[img_name] = label
                json.dump(labels, file, indent=4, ensure_ascii=False)

    def _det(
        self,
        img_output_path: str,
        img_path: str,
        gt: list
    ) -> tuple[str, dict]:
        img = cv2.imread(img_path)
        _, img_name = os.path.split(img_path)

        polygons = []
        for (_, bbox) in gt:
            x1, y1, x2, y2 = tuple(bbox)
            polygons.append([[x1, y1],[x1, y2],[x2, y2],[x2, y1]])

        cv2.imwrite(os.path.join(img_output_path, img_name), img)
        result = (
            img_name,
            dict(
                img_dimensions = (img.shape[1], img.shape[0]),
                img_hash = hashlib.sha256(img).hexdigest(),
                polygons = polygons
            )
        )

        return result
    
    def _rec(
        self,
        img_output_path: str,
        img_path: str,
        gt: list
    ) -> list[tuple[str, str]]:
        img = cv2.imread(img_path)

        _, img_name = os.path.split(img_path)

        result = []
        for i, (text, bbox) in enumerate(gt):
            crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            crop_name = img_name.replace(".", f"-{i}.")
            cv2.imwrite(os.path.join(img_output_path, crop_name), crop)

            result.append((crop_name, text))
        
        return result