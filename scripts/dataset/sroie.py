from .dataset import Dataset
from datasets import load_dataset

CONFIG = {
    "sroie": "darentang/sroie"
}


class SROIE(Dataset):
    def __init__(
        self,
        config: dict
    ) -> None:
        super().__init__(config)

    def download(self):
        super().download()

        for split in ["train", "test"]:
            for sample in load_dataset(self.config[self._current], split=split):
                words = sample["words"]
                bboxes = sample["bboxes"]
                #ner_tags = sample["ner_tags"]
                image_path:str = sample["image_path"]

                image_name = image_path.split("/")[-1]
                file_name = image_name.replace(".jpg", ".txt")

                file = open(f"{self.path()}/{split}/{file_name}","w")
                file_content = []
                for word, bbox in zip(words, bboxes):
                    x1, y1, x2, y2 = tuple(bbox)
                    if (x1 < x2 and y1 < y2):
                        file_content.append(f"{word['text']}\t{bbox}\n")
                file.writelines(file_content)
                file.close()