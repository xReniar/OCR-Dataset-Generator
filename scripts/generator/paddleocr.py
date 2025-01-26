from generator import Generator
import os
import json


class PaddleOCRGenerator(Generator):
    def __init__(
        self,
        datasets: list,
        transforms = None
    ) -> None:
        super().__init__(
            datasets,
            transforms
        )

    def generate_det_data(self):
        super().generate_det_data()

        # images copy step
        for split in ["train", "test"]:
            label_file = open(f"../../output/{self.name()}-det/{split}_label.txt","w")
            for dataset in self.datasets:
                current_path = f"../../data/{dataset}"
                imgs_dir = sorted(os.listdir(f"{current_path}/images"))
                extension_map = {}
                for img in imgs_dir:
                    name, _ = img.split(".")
                    extension_map[f"{name}.txt"] = img

                label_dir = os.listdir(f"{current_path}/{split}")

                # copy images
                self.copy_file(
                    sorted(label_dir),
                    extension_map,
                    current_path,
                    f"../../output/{self.name()}-det/{split}"
                )

                # labels creation
                for label in label_dir:
                    img_name = extension_map[label]

                    text_file = open(f"{current_path}/{split}/{label}")
                    annotations = []
                    for (text, bbox) in self.read_rows(text_file):
                        x1, y1, x2, y2 = bbox
                        annotations.append(dict(
                            transcription=text,
                            points = [[x1, y1],[x2, y1],[x2, y2],[x1, y2]]
                        ))
                    label_file.write(f"{split}/{img_name}\t{json.dumps(annotations)}\n")
                
            label_file.close()
                    

    def generate_rec_data(self):
        super().generate_rec_data()