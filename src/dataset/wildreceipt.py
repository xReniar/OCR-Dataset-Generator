from .dataset import OnlineDataset
import multiprocessing
import tarfile
import requests
import shutil
import json
import os


CONFIG = {
    "wildreceipt": "https://download.openmmlab.com/mmocr/data/wildreceipt.tar"
}

class WILDRECEIPT(OnlineDataset):
    def __init__(   
        self,
        config
    ) -> None:
        super().__init__(config)

    @staticmethod
    def get_bbox(bbox: list[int]):
        bbox = list(map(lambda x: int(x), bbox))
        xm, ym, xM, yM = bbox[0], bbox[1], 0, 0

        for i in range(0, len(bbox), 2):
            coord = bbox[i:i+2]
            xm = min(xm, coord[0])
            ym = min(ym, coord[1])
            xM = max(xM, coord[0])
            yM = max(yM, coord[1])

        return [xm, ym, xM, yM]

    def process_data(
        self,
        file_name_path: str,
        index: int,
        split: str,
        annotations: list
    ) -> None:
        file_name = f"wildreceipt_{split}_img_{index}.jpeg"
        shutil.move(
            os.path.join(self.path(), "wildreceipt", file_name_path),
            os.path.join(self.path(), split, "images", file_name)
        )

        file_id = file_name.split(".")[0]
        with open(os.path.join(self.path(), split, "labels", f"{file_id}.txt"), "w") as file:
            for annotation in annotations:
                box = self.get_bbox(annotation["box"])
                text = annotation["text"]
                if len(text) > 0 and ("\t" not in text):
                    file.write(f"{text}\t{box}\n") 

    def _download(self):
        tar_path = os.path.join(self.path(), "wildreceipt.tar")

        with requests.get(self.config[self._current], stream=True) as response:
            response.raise_for_status()
            with open(tar_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192 * 4):
                    file.write(chunk)
        
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(self.path())
        os.remove(tar_path)
        
        extract_path = tar_path.split(".")[0]

        args = []
        for split in ["train", "test"]:
            labels = open(os.path.join(extract_path, f"{split}.txt"), "r").readlines()
            labels = list(map(lambda row: row.split("\n")[0], labels))

            for index, instance in enumerate(labels):
                label = json.loads(instance)

                args.append((
                    label["file_name"],
                    index,
                    split,
                    label["annotations"]
                ))

        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            pool.starmap(self.process_data, args)
        
        shutil.rmtree(os.path.join(self.path(), "wildreceipt"))