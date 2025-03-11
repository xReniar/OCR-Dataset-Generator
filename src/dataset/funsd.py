from .dataset import Dataset
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

    def _download(self) -> None:
        response = requests.get(self.config[self._current])
        zip_fn = self.config[self._current].split("/")[-1]
        with open(f"{self.path()}/{zip_fn}","wb") as f:
            f.write(response.content)

        zip_path = os.path.join(self.path(), zip_fn)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.path())

        shutil.rmtree(os.path.join(self.path(), "__MACOSX"))
        os.remove(zip_path)

        # move images inside 'images' folder
        for split in ["train", "test"]:
            src_folder = os.path.join(self.path(), "dataset", f"{split}ing_data", "images")
            dst_folder = os.path.join(self.path(), split, "images")

            for item in os.listdir(src_folder):
                shutil.move(
                    os.path.join(src_folder, item),
                    os.path.join(dst_folder, item)
                )

        # creating annotations
        for split in ["train", "test"]:
            src_folder = os.path.join(self.path(), "dataset", f"{split}ing_data", "annotations")
            dst_folder = os.path.join(self.path(), split, "labels")

            for annotation_file in os.listdir(src_folder):
                file_path = os.path.join(dst_folder, annotation_file.replace("json", "txt"))
                file = open(file_path,"w")
                annotation:list = json.load(os.path.join(src_folder, annotation_file), "r")["form"]
                
                file_content = []
                for label in annotation:
                    for word in label["words"]:
                        file_content.append(f"{word['text']}\t{word['box']}\n")
                file.writelines(file_content)
                file.close()

        shutil.rmtree(os.path.join(self.path(), "dataset"))