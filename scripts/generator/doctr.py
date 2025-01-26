from .generator import Generator
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

    def generate_det_data(self):
        super().generate_det_data()

        for split in ["train", "test"]:
            labels = {}
            os.makedirs(f"{self._root_path}/{split}/images", exist_ok=True)
            for dataset in self.datasets:
                current_path = f"data/{dataset}"

                imgs_dir = sorted(os.listdir(f"{current_path}/images"))
                extension_map = self.extension_map(imgs_dir)
                label_dir = sorted(os.listdir(f"{current_path}/{split}"))

                # images creations
                self.copy_file(
                    label_dir,
                    extension_map,
                    current_path,
                    f"{self._root_path}/{split}/images",
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
            with open(f"{self._root_path}/{split}/labels.json","w") as file:
                json.dump(labels,file, indent=4)


    def generate_rec_data(self):
        super().generate_rec_data()

        for split in ["train", "test"]:
            labels = {}
            os.makedirs(f"{self._root_path}/{split}/images", exist_ok=True)
            for dataset in self.datasets:
                current_path = f"data/{dataset}"

                imgs_dir = sorted(os.listdir(f"{current_path}/images"))
                extension_map = self.extension_map(imgs_dir)

                for label in sorted(os.listdir(f"{current_path}/{split}")):
                    text_file = open(f"{current_path}/{split}/{label}")
                    img = Image.open(f"{current_path}/images/{extension_map[label]}")
                    for index, (text, bbox) in enumerate(self.read_rows(text_file)):
                        crop_name = extension_map[label].replace(".",f"-{index}.")
                        img.crop(bbox).save(f"{self._root_path}/{split}/images/{crop_name}")
                        labels[crop_name] = text
                    img.close()
                    text_file.close()

            with open(f"{self._root_path}/{split}/labels.json","w") as file:
                json.dump(labels,file, indent=4)