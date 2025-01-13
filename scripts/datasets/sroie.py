from datasets import Dataset


class SROIE(Dataset):
    def __init__(self):
        super().__init__(
            local_datasets = [],
            online_datasets = {
                "sroie": ""
            }
        )

    def download_images():
        pass