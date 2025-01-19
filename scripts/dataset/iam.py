from dataset import Dataset
from datasets import load_dataset


class IAM(Dataset):
    def __init__(self):
        super().__init__(
            local_datasets = [],
            online_datasets = {
                "iam": "hf url"
            }
        )

    def download_data(self, dataset:str | None = None) -> None:
        super().download_data(dataset)