from .generator import Generator
from ..utils.image import open_image
from ..dataloader import Dataloader
from pathlib import Path
import json
import os
import cv2


class MMOCRGenerator(Generator):
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
            img_output_path = os.path.join(root_path, "imgs", split)
            os.makedirs(img_output_path, exist_ok=True)

            results = self.run_process(img_output_path, dataloader, task, split)

            with open(os.path.join(root_path, f"{split}.json"), "w") as file:
                labels = {}
                if task == "Detection":
                    labels = dict(
                        metainfo = dict(
                            dataset_type = "TextDetDataset",
                            task_name = "textdet",
                            category = [dict(id = 0,name = "text")]
                        ),
                        data_list = results
                    )
                if task == "Recognition":
                    results = [item for sublist in results for item in sublist]
                    labels = dict(
                        metainfo = dict(
                            dataset_type = "TextRecogDataset",
                            task_name = "textrecog"
                        ),
                        data_list = results
                    )
                json.dump(labels, file, indent=4, ensure_ascii=False)
                
    def _det(
        self,
        img_output_path: str,
        img_path: str,
        gt: list,
        transform: tuple[str, callable] = (None, None)
    ) -> dict:
        img = open_image(img_path, transform[1])

        _, img_name = os.path.split(img_path)
        if transform[0] is not None:
            img_name = img_name.replace(".", f"-{transform[0]}.")
        
        cv2.imwrite(os.path.join(img_output_path, img_name), img)

        split = Path(img_output_path).parts[-1]
        instances = []
        for (_, bbox) in gt:
            x1, y1, x2, y2 = list(map(lambda point: float(point), bbox))
            instances.append(dict(
                polygon = [x1, y1, x2, y1, x2, y2, x1, y2],
                bbox = [x1, y1, x2, y2],
                bbox_label = 0,
                ignore = False
            ))
        
        result = dict(
            instances = instances,
            img_path = os.path.join("imgs", split, img_name),
            width = img.shape[1],
            height = img.shape[0]
        )

        return result

    def _rec(
        self,
        img_output_path: str,
        img_path: str,
        gt: list,
        transform: tuple[str, callable] = (None, None)
    ) -> list[dict]:
        img = open_image(img_path, transform[1])

        _, img_name = os.path.split(img_path)
        split = Path(img_output_path).parts[-1]

        data_list = []
        for i, (text, bbox) in enumerate(gt):
            crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if transform[0] is not None:
                crop_name = img_name.replace(".", f"-{i}-{transform[0]}.")
            else:
                crop_name = img_name.replace(".", f"-{i}.")
                
            cv2.imwrite(os.path.join(img_output_path, crop_name), crop)

            data_list.append(dict(
                instances = [{"text": text}],
                img_path = os.path.join("imgs", split, crop_name)
            ))
        
        return data_list