from .dataset import OnlineDataset
from datasets import load_dataset
from PIL import Image
import multiprocessing
import gdown
import zipfile
import os
import shutil

CONFIG = {
    "sroie": [
        "https://drive.google.com/uc?id=1ZyxAw1d-9UvhgNLGRvsJK4gBCMf0VpGD",
        "darentang/sroie"
    ]
}


class SROIE(OnlineDataset):
    def __init__(
        self,
        config: dict
    ) -> None:
        super().__init__(config)

    def get_original_bbox(self, bbox, img_size:tuple):
        width, height = img_size
        return [
            int(width * bbox[0] / 1000),
            int(height * bbox[1] / 1000),
            int(width * bbox[2] / 1000),
            int(height * bbox[3] / 1000),
        ]
    
    def process_data(
        self,
        words: str,
        bboxes: list[int],
        img_name: str,
        split: str
    ) -> None:
        current_path = os.path.join("data", "sroie", split)
        img_id, _ = tuple(img_name.split(".")) 
        with open(os.path.join(current_path, "labels", f"{img_id}.txt"), "w") as file:
            img_path = os.path.join(current_path, "images", img_name)
            img = Image.open(img_path)

            lines = []
            for (word, bbox) in list(zip(words, bboxes)):
                lines.append(f"{word}\t{self.get_original_bbox(bbox, (img.width, img.height))}\n")
            file.writelines(lines)

    def _download(self):
        # download images
        zip_path = os.path.join(self.path(), "sroie.zip")
        gdown.download(self.config[self.__str__()][0], zip_path, quiet=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.path())

        for split in ["train", "test"]:
            src_folder = os.path.join(self.path(), "sroie", split, "images")
            dst_folder = os.path.join(self.path(), split, "images")
            for img_filename in os.listdir(src_folder):
                shutil.move(
                    os.path.join(src_folder, img_filename),
                    os.path.join(dst_folder, img_filename)
                )
        
        os.remove(zip_path)
        shutil.rmtree(os.path.join(self.path(), "__MACOSX"))
        shutil.rmtree(os.path.join(self.path(), "sroie"))
        
        # download labels
        data = []
        for split in ["train", "test"]:
            for sample in load_dataset(self.config[self.__str__()][1], split=split):
                image_path = sample["image_path"]

                data.append((
                    sample["words"],
                    sample["bboxes"],
                    image_path.split("/")[-1],
                    image_path.split("/")[-3]
                ))

        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            pool.starmap(self.process_data, data)