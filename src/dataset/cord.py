from .dataset import Dataset
from datasets import load_dataset
import os
import json


CONFIG = {
    "cord": "naver-clova-ix/cord-v2"
}

class CORD(Dataset):
    def __init__(
        self,
        config
    ) -> None:
        super().__init__(config)

    @staticmethod
    def get_bbox(quad: dict) -> list[int]:
        xm, ym, xM, yM = int(quad["x1"]), int(quad["y1"]), 0, 0
        for key in quad.keys():
            if key[0] == "x":
                xm = min(xm, quad[key])
                xM = max(xM, quad[key])
            if key[0] == "y":
                ym = min(ym, quad[key])
                yM = max(yM, quad[key])

        return [xm, ym, xM, yM]

    def _download(self):
        for split in ["train", "test"]:
            for sample in load_dataset(self.config[self._current], split=split):
                gt = json.loads(sample["ground_truth"])
                img_name:int = gt["meta"]["image_id"]
                image = sample["image"]
                labels: list = gt["valid_line"]

                new_fn = f"img_{img_name}"
                with open(os.path.join(self.path(), split, "labels", f"{new_fn}.txt"), "w") as file:
                    for label in labels:
                        words = label["words"]
                        for word in words:
                            bbox = self.get_bbox(word["quad"])
                            text = word["text"]
                            file.write(f"{text}\t{bbox}\n")

                image.save(os.path.join(self.path(), split, "images", f"{new_fn}.png"))
                image.close()