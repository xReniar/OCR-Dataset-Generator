from .dataset import Dataset
from datasets import load_dataset
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

    def download(self):
        super().download()

        # download images
        zip_path = f"{self.path()}/sroie.zip"
        gdown.download(self.config[self._current][0], zip_path, quiet=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.path())

        for split in ["train", "test"]:
            src_folder = f"{self.path()}/sroie/{split}/images"
            dst_folder = f"{self.path()}/images"
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

                file = open(f"{self.path()}/{split}/{file_name}","w")
                file_content = []
                for word, bbox in zip(words, bboxes):
                    x1, y1, x2, y2 = tuple(bbox)
                    if (x1 < x2 and y1 < y2):
                        file_content.append(f"{word}\t{bbox}\n")
                file.writelines(file_content)
                file.close()