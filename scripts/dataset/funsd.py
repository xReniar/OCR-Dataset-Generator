from dataset import Dataset
import requests
import zipfile
import os
import shutil
import json

# implement something that ignores some characters
class FUNSD(Dataset):
    def __init__(self):
        super().__init__(
            local_datasets = [],
            online_datasets = {
                "funsd": "https://guillaumejaume.github.io/FUNSD/dataset.zip"
            }
        )

    def download_data(self, dataset:str | None = None) -> None:
        super().download_data(dataset)

        response = requests.get(self._current_link)
        zip_fn = self._current_link.split("/")[-1]
        with open(f"{self.get_path(dataset)}/{zip_fn}","wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(f"{self.get_path(dataset)}/{zip_fn}", "r") as zip_ref:
            zip_ref.extractall(self.get_path(dataset))

        shutil.rmtree(os.path.join(self.get_path(dataset), "__MACOSX"))
        os.remove(os.path.join(self.get_path(dataset), zip_fn))

        # move images inside 'images' folder
        for split in ["train", "test"]:
            src_folder = f"{self.get_path(dataset)}/dataset/{split}ing_data/images"
            dst_folder = f"{self.get_path(dataset)}/images"

            for item in os.listdir(src_folder):
                shutil.move(
                    os.path.join(src_folder, item),
                    os.path.join(dst_folder, item)
                )

        # creating annotations
        for split in ["train", "test"]:
            src_folder = f"{self.get_path(dataset)}/dataset/{split}ing_data/annotations"
            dst_folder = f"{self.get_path(dataset)}/{split}"

            for annotation_file in os.listdir(src_folder):
                file = open(f"{self.get_path(dataset)}/{split}/{annotation_file.split('.')[0]}.txt","w")
                annotation:list = json.load(open(f"{src_folder}/{annotation_file}", "r"))["form"]
                
                for label in annotation:
                    for word in label["words"]:
                        file.write(f"{word['text']}\t{word['box']}\n")

                file.close()

        shutil.rmtree(os.path.join(self.get_path(dataset), "dataset"))