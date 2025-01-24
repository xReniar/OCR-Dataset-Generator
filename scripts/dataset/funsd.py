from dataset import Dataset
import requests
import zipfile
import shutil
import json
import os


CONFIG = {
    "funsd": "https://guillaumejaume.github.io/FUNSD/dataset.zip"
}

class FUNSD(Dataset):
    def __init__(
        self,
        config: dict
    ) -> None:
        super().__init__(config)

    def _download(self):
        super()._download()

        response = requests.get(CONFIG[self._current])
        zip_fn = CONFIG[self._current].split("/")[-1]
        with open(f"{self.path()}/{zip_fn}","wb") as f:
            f.write(response.content)

        with zipfile.ZipFile(f"{self.path()}/{zip_fn}", "r") as zip_ref:
            zip_ref.extractall(self.path())

        shutil.rmtree(os.path.join(self.path(), "__MACOSX"))
        os.remove(os.path.join(self.path(), zip_fn))

        # move images inside 'images' folder
        for split in ["train", "test"]:
            src_folder = f"{self.path()}/dataset/{split}ing_data/images"
            dst_folder = f"{self.path()}/images"

            for item in os.listdir(src_folder):
                shutil.move(
                    os.path.join(src_folder, item),
                    os.path.join(dst_folder, item)
                )

        # creating annotations
        for split in ["train", "test"]:
            src_folder = f"{self.path()}/dataset/{split}ing_data/annotations"
            dst_folder = f"{self.path()}/{split}"

            for annotation_file in os.listdir(src_folder):
                file = open(f"{self.path()}/{split}/{annotation_file.split('.')[0]}.txt","w")
                annotation:list = json.load(open(f"{src_folder}/{annotation_file}", "r"))["form"]
                
                for label in annotation:
                    for word in label["words"]:
                        file.write(f"{word['text']}\t{word['box']}\n")

                file.close()

        shutil.rmtree(os.path.join(self.path(), "dataset"))