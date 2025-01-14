from dataset import Dataset


class SROIE(Dataset):
    def __init__(self):
        super().__init__(
            local_datasets = [],
            online_datasets = {
                "sroie": ""
            }
        )

    def download_images(self, dataset:str | None = None):
        super().download_images(dataset)
        link:str = self.online_datasets[dataset if dataset != None else self.get_class_name()]

        # to do