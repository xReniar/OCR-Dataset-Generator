from dataset import Dataset


class XFUND(Dataset):
    def __init__(self):
        super().__init__(
            local_datasets = [],
            online_datasets = {
                "de": "",
                "es": "",
                "fr": "",
                "it": "",
                "ja": "",
                "pt": "",
                "zh": "",
            }
        )

    def download_images(dataset:str | None = None):
        pass