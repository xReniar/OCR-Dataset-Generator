from .dataset import Dataset
import requests
import zipfile
import os
import shutil


CONFIG = {
    "iam": {
        "images-link": "https://www.kaggle.com/api/v1/datasets/download/naderabdalghani/iam-handwritten-forms-dataset",
        "split-link": "https://fki.tic.heia-fr.ch/static/zip/largeWriterIndependentTextLineRecognitionTask.zip",
        "labels-link": "nibinv23/iam-handwriting-word-database"
    }
}

class IAM(Dataset):
    def __init__(
        self,
        config: dict
    ) -> None:
        super().__init__(config)

    def _download(self):
        # download images
        zip_path = os.path.join(self.path(), "dataset.zip")
        with requests.get(
            self.config[self._current]["images-link"],
            allow_redirects=True,
            stream=True
        ) as response:
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192 * 4):
                    if chunk:
                        f.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.path())

        os.remove(zip_path)
        os.remove(os.path.join(self.path(), "__notebook_source__.ipynb"))

        data_folder_path = os.path.join(self.path(), "data")
        for folder in os.listdir(data_folder_path):
            folder_path = os.path.join(self.path(), "data", folder)
            for fn in os.listdir(folder_path):
                shutil.move(
                    os.path.join(folder_path, fn),
                    os.path.join(data_folder_path)
                )
            os.removedirs(folder_path)

        # download split
        zip_path = os.path.join(self.path(), "split.zip")
        response = requests.get(self.config[self._current]["split-link"])

        with open(zip_path, "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.path())
        os.remove(zip_path)
        os.remove(os.path.join(self.path(), "LargeWriterIndependentTextLineRecognitionTask.txt"))
        os.remove(os.path.join(self.path(), "testset.txt"))

        train_file = list(map(lambda x: x.split("\n")[0], open(os.path.join(self.path(), "trainset.txt")).readlines()))
        val_file_1 = list(map(lambda x: x.split("\n")[0], open(os.path.join(self.path(), "validationset1.txt")).readlines()))
        val_file_2 = list(map(lambda x: x.split("\n")[0], open(os.path.join(self.path(), "validationset2.txt")).readlines()))

        train = set()
        for line_id in train_file:
            a, b, _ = line_id.split("-")
            train.add(f"{a}-{b}")
        
        val = set()
        for line_id in val_file_1 + val_file_2:
            a, b, _ = line_id.split("-")
            val.add(f"{a}-{b}")

        os.remove(os.path.join(self.path(), "trainset.txt"))
        os.remove(os.path.join(self.path(), "validationset1.txt"))
        os.remove(os.path.join(self.path(), "validationset2.txt"))