from .dataset import Dataset
from datasets import load_dataset
import cv2
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


class SROIE(Dataset):
    def __init__(
        self,
        config: dict
    ) -> None:
        super().__init__(config)

    def get_original_bbox(self, bbox, img_path:str):
        img = cv2.imread(img_path)
        width, height = img.shape[1], img.shape[0]
        return [
            int(width * bbox[0] / 1000),
            int(height * bbox[1] / 1000),
            int(width * bbox[2] / 1000),
            int(height * bbox[3] / 1000),
        ]

    def _download(self):
        # download images
        zip_path = os.path.join(self.path(), "sroie.zip")
        gdown.download(self.config[self._current][0], zip_path, quiet=True)

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
        for split in ["train", "test"]:
            for sample in load_dataset(self.config[self._current][1], split=split):
                words = sample["words"]
                bboxes = sample["bboxes"]
                #ner_tags = sample["ner_tags"]
                image_path:str = sample["image_path"]

                image_name = image_path.split("/")[-1]
                file_name = image_name.replace(".jpg", ".txt")

                file = open(os.path.join(self.path(), split, "labels", file_name),"w")
                file_content = []
                for word, bbox in zip(words, bboxes):
                    img_path = os.path.join(self.path(), split, "images", image_name)
                    bbox = self.get_original_bbox(bbox, img_path=img_path)
                    file_content.append(f"{word}\t{bbox}\n")
                file.writelines(file_content)
                file.close()