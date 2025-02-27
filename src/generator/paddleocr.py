from .generator import Generator
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

    def _generate_det_data(self):
        self._root_path = os.path.join(self._root_path, "Detection")
        os.makedirs(self._root_path, exist_ok=True)

        # images copy step
        for split in ["train", "test"]:
            os.makedirs(os.path.join(self._root_path, split), exist_ok=True)
            label_file = open(f"{self._root_path}/{split}_label.txt","w")
            for dataset in self.datasets:
                current_path = f"data/{dataset}"
                extension_map = self.extension_map(sorted(os.listdir(f"{current_path}/images")))

                label_dir = sorted(os.listdir(f"{current_path}/{split}"))

                self.copy_file(
                    sorted(label_dir),
                    extension_map,
                    current_path,
                    f"{self._root_path}/{split}"
                )

                label_content = []
                # labels creation
                for label in label_dir:
                    img_name = extension_map[label]

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
                    

    def _generate_rec_data(self):
        self._root_path = os.path.join(self._root_path, "Recognition")
        os.makedirs(self._root_path, exist_ok=True)
        
        for split in ["train", "test"]:
            os.makedirs(os.path.join(self._root_path, split), exist_ok=True)
            label_file = open(f"{self._root_path}/{split}_label.txt","w")
            for dataset in self.datasets:
                current_path = f"data/{dataset}"
                imgs_dir = sorted(os.listdir(f"{current_path}/images"))
                extension_map = self.extension_map(imgs_dir)
                label_dir = sorted(os.listdir(f"{current_path}/{split}"))

                label_content = []
                for label in label_dir:
                    img = Image.open(f"{current_path}/images/{extension_map[label]}")
                    for index, (text, bbox) in enumerate(self.read_rows(f"{current_path}/{split}/{label}")):
                        crop_name = extension_map[label].replace(".",f"-{index}.")
                        img.crop(bbox).save(f"{self._root_path}/{split}/{crop_name}")
                        label_content.append(f"{split}/{crop_name}\t{text}\n")
                label_file.writelines(label_content)
            label_file.close()