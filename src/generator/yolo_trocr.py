from .generator import Generator
from ..dataloader import Dataloader
from ..utils.image import open_image
import cv2
import os


class YOLOTrOCRGenerator(Generator):
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
            if task == "Detection":
                os.makedirs(os.path.join(img_output_path, "images"), exist_ok=True)
                os.makedirs(os.path.join(img_output_path, "labels"), exist_ok=True)

            results = self.run_process(img_output_path, dataloader, task, split)

            if task == "Recognition":
                with open(os.path.join(root_path, f"{split}.txt"), "w") as file:
                    results = [item for sublist in results for item in sublist]
                    file.writelines(results)

    def _det(
        self,
        img_output_path: str,
        img_path: str,
        gt: list,
        transform: tuple[str, callable] = (None, None)
    ) -> None:
        img = open_image(img_path, transform[1])
        _, img_name = os.path.split(img_path)
        if transform[0] is not None:
            img_name = img_name.replace(".", f"-{transform[0]}.")

        img_id = img_name.split(".")[0]

        cv2.imwrite(os.path.join(img_output_path, "images" ,img_name), img)
        width, height = img.shape[1], img.shape[0]

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

    def _rec(
        self,
        img_output_path: str,
        img_path: str,
        gt: list,
        transform: tuple[str, callable] = (None, None)
    ) -> list[str]:
        img = open_image(img_path, transform[1])
        _, img_name = os.path.split(img_path)

        _, img_name = os.path.split(img_path)
        result = []
        for i, (text, bbox) in enumerate(gt):
            crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if transform[0] is not None:
                crop_name = img_name.replace(".", f"-{i}-{transform[0]}.")
            else:
                crop_name = img_name.replace(".", f"-{i}.")
                
            cv2.imwrite(os.path.join(img_output_path, crop_name), crop)
            result.append(f"{crop_name} {text}\n")
        
        return result