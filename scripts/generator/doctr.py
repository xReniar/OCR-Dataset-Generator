from generator import Generator
from PIL import Image
import hashlib
import json
import os


class DoctrGenerator(Generator):
    def __init__(
        self,
        datasets:list,
        transforms = None
    ) -> None:
        super().__init__(
            datasets,
            transforms
        )

    def generate_det_data(self):
        super().generate_det_data()

        labels = {}

        for split in ["train", "test"]:
            for dataset in self.datasets:
                current_path = f"../../data/{dataset}"

                imgs_dir = os.listdir(f"{current_path}/images")
                imgs_dir.sort()

                extension_map = {}
                for img in imgs_dir:
                    name, _ = img.split(".")
                    extension_map[f"{name}.txt"] = img

                os.makedirs(f"../../output/doctr-det/{split}/images", exist_ok=True)
                label_dir = os.listdir(f"{current_path}/{split}")
                label_dir.sort()

                # images creations
                self.copy_file(
                    label_dir,
                    extension_map,
                    current_path,
                    f"../../output/{self.name()}-det/{split}/images",
                )

                # labels creation
                for label in label_dir:
                    img_name = extension_map[label]

                    # creating label instance for labels.json
                    img = Image.open(f"{current_path}/images/{img_name}")
                    polygons = []
                    text_file = open(f"{current_path}/{split}/{label}")
                    for (_, bbox) in self.read_rows(text_file):
                        x1, y1, x2, y2 = bbox
                        polygons.append([[x1, y1],[x2, y1],[x2, y2],[x1, y2]])
                    text_file.close()
                    labels[f"{img_name}"] = dict(
                        img_dimensions = (img.width, img.height),
                        img_hash = hashlib.sha256(img.tobytes()).hexdigest(),
                        polygons = polygons
                    )
                    img.close()
            with open(f"../../output/doctr-det/{split}/labels.json","w") as file:
                json.dump(labels,file, indent=4)


    def generate_rec_data(self):
        super().generate_rec_data()

        for split in ["train", "test"]:
            for dataset in self.datasets:
                current_path = f"../../data/{dataset}"

                imgs_dir = os.listdir(f"{current_path}/images")
                imgs_dir.sort()

                os.makedirs(f"../../output/doctr-rec/{split}/images", exist_ok=True)
                label_dir = os.listdir(f"{current_path}/{split}")
                label_dir.sort()