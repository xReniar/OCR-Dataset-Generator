from dataset import Dataset
from datasets import load_dataset

CONFIG = {
    "sroie": [
        "",
        "darentang/sroie"
    ]
}


class SROIE(Dataset):
    def __init__(
        self,
        config: dict
    ) -> None:
        super().__init__(config)

    def _download(self):
        super()._download()

        for split in ["train", "test"]:
            for sample in load_dataset(CONFIG[self._current][1], split=split):
                words = sample["words"]
                bboxes = sample["bboxes"]
                #ner_tags = sample["ner_tags"]
                image_path:str = sample["image_path"]

                image_name = image_path.split("/")[-1]
                file_name = image_name.replace(".jpg", ".txt")

                file = open(f"{self.path()}/{split}/{file_name}","w")
                for word, bbox in zip(words, bboxes):
                    file.write(f"{word}\t{bbox}\n")
                file.close()