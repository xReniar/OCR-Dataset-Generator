from abc import ABC
from .utils import reader
import multiprocessing
import os


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
        self._workers = os.cpu_count()
        
        self.__load_data__()

    def __load_data__(self) -> None:
        for split in ["train", "test"]:
            root_paths = [os.path.join("data", dataset, split, "labels") for dataset in self._datasets]

            full_paths = []
            for path in root_paths:
                full_paths += [os.path.join(path, fn) for fn in os.listdir(path)]

            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                labels = pool.map(reader.read_label, full_paths)

            self.data[split] = labels
    
    def __filter__(
        self,
        data: list
    ) -> None:
        pass