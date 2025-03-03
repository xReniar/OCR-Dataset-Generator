from abc import ABC, abstractmethod


class Dataloader(ABC):
    def __init__(
        self,
        transforms: dict,
        datasets: list[str]
    ) -> None:
        super().__init__()

        self.transforms = transforms
        self.datasets = datasets

        self.train_data = []
        self.test_data = []

        #self.load_data()

    @abstractmethod
    def load_data(self, split:str):
        pass
    
