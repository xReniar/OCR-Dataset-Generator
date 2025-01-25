from abc import ABC, abstractmethod
import os


class Generator(ABC):
    base_path = "../../output"

    def __init__(
        self,
        datasets : list,
        transforms,
    ) -> None:
        super().__init__()

        self.datasets:list = datasets
        self.transforms = transforms

    def name(
        self
    ) -> str:
        return self.__class__.__name__.lower().replace("generator","")

    @abstractmethod
    def generate_det_data(
        self
    ) -> None:
        os.makedirs(
            os.path.join(self.base_path, self.name() + "-det"),
            exist_ok=True
        )
    
    @abstractmethod
    def generate_rec_data(
        self
    ) -> None:
        os.makedirs(
            os.path.join(self.base_path, self.name() + "-rec"),
            exist_ok=True
        )