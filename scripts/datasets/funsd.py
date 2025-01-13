from dataset import Dataset


class FUNSD(Dataset):
    def __init__(self):
        super().__init__(
            local_datasets = [],
            online_datasets = {
                "funsd": ""
            }
        )

    def download_images():
        pass