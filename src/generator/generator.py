from abc import ABC, abstractmethod
from ..dataloader import Dataloader
import multiprocessing
import os


class Generator(ABC):
    base_path = "output"

    def __init__(
        self,
        test_name: str,
        datasets : list[str],
        transforms
    ) -> None:
        super().__init__()

        new_datasets = []
        for dataset in datasets:
            element = dataset
            if "-" in element:
                dataset_root = dataset.split("-")[0]
                element = os.path.join(dataset_root, element)
            new_datasets.append(element)

        self.test_name = test_name
        self.datasets:list[str] = new_datasets
        self.transforms = transforms

    def name(
        self
    ) -> str:
        return self.__class__.__name__.lower().replace("generator","")

    def generate_data(self, tasks:dict):
        self.root_path = os.path.join(self.base_path,f"{self.test_name}-{self.name()}")
        os.makedirs(self.root_path, exist_ok=True)

        dataloader = Dataloader(
            self.transforms,
            self.datasets
        )

        if tasks["det"] == "y":
            self._generate(dataloader, "Detection", self._det)
        if tasks["rec"] == "y":
            self._generate(dataloader, "Recognition", self._rec)

    @abstractmethod
    def _generate(self, dataloader: Dataloader, task: str, process) -> None:
        pass

    @abstractmethod
    def _det(self, img_output_path: str, img_path: str, gt: list) -> None:
        pass
    
    @abstractmethod
    def _rec(self, img_output_path: str, img_path: str, gt: list) -> None:
        pass