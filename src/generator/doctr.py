from .generator import Generator
from ..dataloader.detLoader import DetDataloader
from ..dataloader.recLoader import RecDataloader
from PIL import Image
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

    def _generate_det_data(self, dataloader: DetDataloader):
        root_path = os.path.join(self.root_path, "Detection")
        os.makedirs(root_path, exist_ok=True)
        
        for split in ["train", "test"]:
            labels = {}
            os.makedirs(f"{root_path}/{split}/images", exist_ok=True)
            for dataset in self.datasets:
                current_path = f"data/{dataset}"

                label_dir = sorted(os.listdir(f"{current_path}/{split}"))

                # images creations
                self.copy_file(
                    label_dir,
                    current_path,
                    f"{root_path}/{split}/images",
                )

                # labels creation
                for label in label_dir:
                    img_name = label.replace(".txt","")

                    # creating label instance for labels.json
                    img = Image.open(f"{current_path}/images/{img_name}")
                    polygons = []
                    for (_, bbox) in self.read_rows(f"{current_path}/{split}/{label}"):
                        x1, y1, x2, y2 = bbox
                        polygons.append([[x1, y1],[x2, y1],[x2, y2],[x1, y2]])
                    labels[f"{img_name}"] = dict(
                        img_dimensions = (img.width, img.height),
                        img_hash = hashlib.sha256(img.tobytes()).hexdigest(),
                        polygons = polygons
                    )
                    img.close()
            with open(f"{root_path}/{split}/labels.json","w") as file:
                json.dump(labels,file, indent=4)

    def _generate_rec_data(self, dataloader: RecDataloader):
        root_path = os.path.join(self.root_path, "Recognition")
        os.makedirs(root_path, exist_ok=True)

        for split in ["train", "test"]:
            labels = {}
            os.makedirs(f"{root_path}/{split}/images", exist_ok=True)
            for dataset in self.datasets:
                current_path = f"data/{dataset}"

                for label in sorted(os.listdir(f"{current_path}/{split}")):
                    img_fn = label.replace(".txt", "")
                    img = Image.open(f"{current_path}/images/{img_fn}")
                    for index, (text, bbox) in enumerate(self.read_rows(f"{current_path}/{split}/{label}")):
                        crop_name = img_fn.replace(".",f"-{index}.")
                        img.crop(bbox).save(f"{root_path}/{split}/images/{crop_name}")
                        labels[crop_name] = text
                    img.close()

            with open(f"{root_path}/{split}/labels.json","w") as file:
                json.dump(labels,file, indent=4, ensure_ascii=False)