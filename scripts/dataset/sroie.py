from dataset import Dataset
from datasets import load_dataset


class SROIE(Dataset):
    def __init__(self):
        super().__init__(
            local_datasets = [],
            online_datasets = {
                "sroie": "darentang/sroie"
            }
        )

    def download_data(self, dataset:str | None = None) -> None:
        super().download_data(dataset)

        for split in ["train", "test"]:
            for sample in load_dataset(self._current_link, split=split):
                words = sample["words"]
                bboxes = sample["bboxes"]
                #ner_tags = sample["ner_tags"]
                image_path:str = sample["image_path"]

                image_name = image_path.split("/")[-1]
                file_name = image_name.replace(".jpg", ".txt")

                file = open(f"{self.get_path(dataset)}/{split}/{file_name}","w")
                for word, bbox in zip(words, bboxes):
                    file.write(f"{word}\t{bbox}\n")
                file.close()