from dataset import Dataset


class IAM(Dataset):
    def __init__(self):
        super().__init__(
            local_datasets = [],
            online_datasets = {
                "htr": "",
                "online": ""
            }
        )

    def download_images():
        pass