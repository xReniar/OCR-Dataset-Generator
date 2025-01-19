from abc import ABC, abstractmethod


class Generator(ABC):
    def __init__(self,
        datasets: tuple,
        task: str,
        output_name: str,
        transforms,
    ) -> None:
        super().__init__()

        self.datasets: tuple = datasets
        self.task: str = task
        self.output_name: str = output_name
        self.transforms = transforms

    @abstractmethod
    def det_generator(self, dataset:str) -> None:
        pass
    
    @abstractmethod
    def rec_generator(self, dataset:str) -> None:
        pass