from .dataset import Dataset
import cv2
import multiprocessing
import requests
import gdown
import zipfile
import os
import shutil

CONFIG = {
    "sroie": [
        "https://drive.google.com/uc?id=1ZyxAw1d-9UvhgNLGRvsJK4gBCMf0VpGD",
        "https://datasets-server.huggingface.co/rows?dataset=darentang%2Fsroie&config=sroie"
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
    
    def process_data(
        self,
        base_url: str,
        split: str,
        offset: str
    ) -> None:
        response = requests.get(f"{base_url}&split={split}&offset={offset}&length=100")
        json_response = response.json()

        current_path = os.path.join("data", "sroie", split)

        for row in json_response["rows"]:
            instance = row["row"]

            words: str = instance["words"]
            bboxes: list[int] = instance["bboxes"]
            img_name: str = instance["image_path"].split("/")[-1]
            img_id, _ = tuple(img_name.split("."))
            img_path = os.path.join(current_path, "images", img_name)

            with open(os.path.join(current_path, "labels", f"{img_id}.txt"), "w") as file:
                lines = []
                for (word, bbox) in list(zip(words, bboxes)):
                    lines.append(f"{word}\t{self.get_original_bbox(bbox, img_path)}\n")
                file.writelines(lines)

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
        size = {
            "train": 626,
            "test": 347
        }
        args = []
        for split in size.keys():
            offset = 0
            while offset < size[split]:
                args.append((
                    self.config[self._current][1],
                    split,
                    offset
                ))
                offset += 100

        pool = multiprocessing.Pool(processes=os.cpu_count())
        _ = pool.starmap(self.process_data, args)
        pool.close()
        pool.join()