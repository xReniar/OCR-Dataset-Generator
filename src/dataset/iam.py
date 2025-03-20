from .dataset import Dataset


CONFIG = {
    "iam": [
        "naderabdalghani/iam-handwritten-forms-dataset",
        "nibinv23/iam-handwriting-word-database"
    ]
}

class IAM(Dataset):
    def __init__(
        self,
        config: dict
    ) -> None:
        super().__init__(config)

    def _download(self):
        pass