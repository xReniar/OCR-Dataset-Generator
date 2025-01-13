from datasets import Dataset


class FUNSD(Dataset):
    def __init__(self):
        super().__init__(
            local_datasets = [],
            online_datasets = []
        )