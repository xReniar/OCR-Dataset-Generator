from abc import ABC, abstractmethod
import ast


class Dataloader(ABC):
    def __init__(
        self,
        transforms: dict,
        datasets: list[str]
    ) -> None:
        super().__init__()

        self._transforms = transforms
        self._datasets = datasets
        self.data = dict(
            train = [],
            test = []
        )

        self._load_data()

    @abstractmethod
    def _load_data(self):
        pass