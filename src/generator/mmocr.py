from .generator import Generator
from ..dataloader import Dataloader
from PIL import Image
from pathlib import Path
import multiprocessing
import os
import json


class MMOCRGenerator(Generator):
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
        process,
    ) -> None:
        root_path = os.path.join(self.root_path, task)
        os.makedirs(root_path, exist_ok=True)

        for split in ["train", "test"]:
            img_output_path = os.path.join(root_path, "imgs", split)
            os.makedirs(img_output_path, exist_ok=True)

            args = [(img_output_path, img_path, gt) for (img_path, gt) in dataloader.data[split]]

            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                results = pool.starmap(process, args)

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
        gt: list
    ) -> dict:
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        _, img_name = os.path.split(img_path)
        img.save(os.path.join(img_output_path, img_name))

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
            width = img.width,
            height = img.height
        )
        img.close()

        return result

    def _rec(
        self,
        img_output_path: str,
        img_path: str,
        gt: list
    ) -> list[dict]:
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')

        _, img_name = os.path.split(img_path)
        split = Path(img_output_path).parts[-1]

        data_list = []
        for i, (text, bbox) in enumerate(gt):
            crop = img.crop(bbox)
            crop_name = img_name.replace(".", f"-{i}.")
            crop.save(os.path.join(img_output_path, crop_name))
            crop.close()

            data_list.append(dict(
                instances = [{"text": text}],
                img_path = os.path.join("imgs", split, img_name)
            ))
        img.close()
        
        return data_list