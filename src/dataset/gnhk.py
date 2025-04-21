from .dataset import OnlineDataset
import gdown
import json
import multiprocessing
import os
import shutil
import zipfile


CONFIG = {
    "gnhk": {
        "train": "https://drive.google.com/uc?id=1LxldTGL6PzZChfPYdneMcXoAqwKrsDjP",
        "test": "https://drive.google.com/uc?id=1Cks3veqnNSUjOdRILeBsTv32w94gLJnK"
    }
}

class GNHK(OnlineDataset):
    def __init__(
            self,
            config
        ) -> None:
        super().__init__(config)

    def process_label(
        self,
        s: str,
        split: str
    ) -> None:
        label = json.loads(s)
        img_name: str = label["source-ref"]
        annotations: list = label["annotations"]["texts"]

        label_name = img_name.replace(".jpg", ".txt")
        with open(os.path.join(self.path(), split, "labels", label_name), "w") as file:
            for annotation in annotations:
                text = annotation["text"]
                polygon = annotation["polygon"]

                xm, ym, xM, yM = polygon[0]["x"], polygon[0]["y"], 0, 0
                for coord in polygon:
                    xm = min(xm, coord["x"])
                    ym = min(ym, coord["y"])
                    xM = max(xM, coord["x"])
                    yM = max(yM, coord["y"])
                bbox = [xm, ym, xM, yM]

                file.write(f"{text}\t{bbox}\n")

    def _download(self):
        # download images
        for split in ["train", "test"]:
            zip_path = os.path.join(self.path(), f"{split}.zip")
            gdown.download(self.config[self._current][split], zip_path, quiet=True)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.path())

            os.remove(os.path.join(self.path(), f"{split}.zip"))

        # setup labels
        for split in ["train", "test"]:
            split_folder_path = os.path.join(self.path(), split)
            split_folder = os.listdir(split_folder_path)

            split_folder.remove(f"{split}.manifest")
            split_folder.remove("labels")
            split_folder.remove("images")

            for img_fn in split_folder:
                shutil.move(
                    os.path.join(split_folder_path, img_fn),
                    os.path.join(split_folder_path, "images")
                )

            labels = list(map(lambda row: row.strip("\n"), open(os.path.join(split_folder_path, f"{split}.manifest"), "r").readlines()))
            labels = [(label, split) for label in labels]

            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                pool.starmap(self.process_label, labels)

            os.remove(os.path.join(split_folder_path, f"{split}.manifest"))