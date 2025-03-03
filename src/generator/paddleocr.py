from .generator import Generator
from ..dataloader.detLoader import DetDataloader
from ..dataloader.recLoader import RecDataloader
from PIL import Image
import os
import json


class PaddleOCRGenerator(Generator):
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

    def _generate_det_data(self, dataloader: DetDataloader):
        root_path = os.path.join(self.root_path, "Detection")
        os.makedirs(root_path, exist_ok=True)

        # images copy step
        for split in ["train", "test"]:
            os.makedirs(os.path.join(root_path, split), exist_ok=True)
            label_file = open(f"{root_path}/{split}_label.txt","w")
            for dataset in self.datasets:
                current_path = f"data/{dataset}"

                label_dir = sorted(os.listdir(f"{current_path}/{split}"))

                self.copy_file(
                    sorted(label_dir),
                    current_path,
                    f"{root_path}/{split}"
                )

                label_content = []
                # labels creation
                for label in label_dir:
                    img_name = label.replace(".txt","")

                    annotations = []
                    for (text, bbox) in self.read_rows(f"{current_path}/{split}/{label}"):
                        x1, y1, x2, y2 = bbox
                        annotations.append(dict(
                            transcription=text,
                            points = [[x1, y1],[x2, y1],[x2, y2],[x1, y2]]
                        ))
                    label_content.append(f"{split}/{img_name}\t{json.dumps(annotations, ensure_ascii=False)}\n")
                label_file.writelines(label_content)
            label_file.close()

    def _generate_rec_data(self, dataloader: RecDataloader):
        root_path = os.path.join(self.root_path, "Recognition")
        os.makedirs(root_path, exist_ok=True)
        
        for split in ["train", "test"]:
            os.makedirs(os.path.join(root_path, split), exist_ok=True)
            label_file = open(f"{root_path}/{split}_label.txt","w")
            for dataset in self.datasets:
                current_path = f"data/{dataset}"
                label_dir = sorted(os.listdir(f"{current_path}/{split}"))

                label_content = []
                for label in label_dir:
                    img_name = label.replace(".txt","")
                    img = Image.open(f"{current_path}/images/{img_name}")
                    for index, (text, bbox) in enumerate(self.read_rows(f"{current_path}/{split}/{label}")):
                        crop_name = img_name.replace(".",f"-{index}.")
                        img.crop(bbox).save(f"{root_path}/{split}/{crop_name}")
                        label_content.append(f"{split}/{crop_name}\t{text}\n")
                label_file.writelines(label_content)
            label_file.close()