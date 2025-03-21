from .dataset import Dataset
import requests
import zipfile
import os
import shutil


CONFIG = {
    "iam": {
        "images-link": "https://www.kaggle.com/api/v1/datasets/download/naderabdalghani/iam-handwritten-forms-dataset",
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

        os.remove(os.path.join(self.path(), "dataset.zip"))
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
