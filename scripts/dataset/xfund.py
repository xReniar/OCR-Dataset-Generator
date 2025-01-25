from .dataset import Dataset
import requests
import zipfile
import os

ROOT = "https://github.com/doc-analysis/XFUND/releases/download/v1.0"

CONFIG = {
    "xfund-de": f"{ROOT}/de.",
    "xfund-es": f"{ROOT}/es.",
    "xfund-fr": f"{ROOT}/fr.",
    "xfund-it": f"{ROOT}/it.",
    "xfund-ja": f"{ROOT}/ja.",
    "xfund-pt": f"{ROOT}/pt.",
    "xfund-zh": f"{ROOT}/zh."
}


class XFUND(Dataset):
    def __init__(
        self,
        config: dict
    ) -> None:
        super().__init__(config)

    def download(self):
        super().download()

        # download and extraction of annotations
        for split in ["train", "val"]:
            folder = "train" if split == "train" else "test"
            annotation = requests.get(f"{self.config[self._current]}{split}.json", allow_redirects=True).json()
            documents:list = annotation["documents"]
            
            for element in documents:
                # contains box and text
                document = element["document"]
                # contains 'fname', 'width' and 'height'
                fname:str = element["img"]["fname"]

                file = open(f"{self.path()}/{folder}/{fname.split('.')[0]}.txt", "w")
                for labels in document:
                    for word in labels["words"]:
                        file.write(f"{word['text']}\t{word['box']}\n")
                file.close()

        # download images
        for split in ["train", "val"]:
            folder = "train" if split == "train" else "test"
            response = requests.get(f"{self.config[self._current]}{split}.zip")
            zip_fn = self.config[self._current].split("/")[-1]
            current_path = f"{self.path()}/{zip_fn}{split}.zip"
            with open(current_path,"wb") as f:
                f.write(response.content)

            with zipfile.ZipFile(current_path, "r") as zip_ref:
                zip_ref.extractall(os.path.join(self.path(), "images"))

            os.remove(current_path)