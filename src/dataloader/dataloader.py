from abc import ABC, abstractmethod
import ast


class Dataloader(ABC):
    def __init__(
        self,
        transforms: dict,
        datasets: list[str]
    ) -> None:
        super().__init__()

        self.transforms = transforms
        self.datasets = datasets
        self.data = {}

        self._load_data()

    def read_label(
        self,
        label_path:str
    ) -> None:
        label_content = []
        with open(label_path, "r") as label:
            label = label.readlines()
            for row in label:
                text, bbox = tuple(row.split("\t"))
                bbox = ast.literal_eval(bbox.strip("\n"))
                label_content.append((text, bbox))

        return label_content

    @abstractmethod
    def _load_data(self):
        pass