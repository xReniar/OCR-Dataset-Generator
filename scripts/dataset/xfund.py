from dataset import Dataset
import requests
import zipfile
import os

URL = "https://github.com/doc-analysis/XFUND/releases/download/v1.0"

class XFUND(Dataset):
    def __init__(self):
        super().__init__(
            local_datasets = [],
            online_datasets = {
                "de": f"{URL}/de.",
                "es": f"{URL}/es.",
                "fr": f"{URL}/fr.",
                "it": f"{URL}/it.",
                "ja": f"{URL}/ja.",
                "pt": f"{URL}/pt.",
                "zh": f"{URL}/zh.",
            }
        )

    def download_data(self, dataset:str | None = None) -> None:
        super().download_data(dataset)

        # download and extraction of annotations
        for split in ["train", "val"]:
            folder = "train" if split == "train" else "test"
            annotation = requests.get(f"{self._current_link}{split}.json", allow_redirects=True).json()
            documents:list = annotation["documents"]
            
            for element in documents:
                # contains box and text
                document = element["document"]
                # contains 'fname', 'width' and 'height'
                fname:str = element["img"]["fname"]

                file = open(f"{self.get_path(dataset)}/{folder}/{fname.split('.')[0]}.txt", "w")
                for labels in document:
                    for word in labels["words"]:
                        file.write(f"{word['text']}\t{word['box']}\n")
                file.close()

        # download images
        for split in ["train", "val"]:
            folder = "train" if split == "train" else "test"
            response = requests.get(f"{self._current_link}{split}.zip")
            zip_fn = self._current_link.split("/")[-1]
            current_path = f"{self.get_path(dataset)}/{zip_fn}{split}.zip"
            with open(current_path,"wb") as f:
                f.write(response.content)

            with zipfile.ZipFile(current_path, "r") as zip_ref:
                zip_ref.extractall(os.path.join(self.get_path(dataset), "images"))

            os.remove(current_path)