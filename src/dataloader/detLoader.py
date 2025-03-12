from .dataloader import Dataloader
from ..utils import reader
import multiprocessing
import os

class DetDataloader(Dataloader):
    def __init__(
        self,
        transforms: dict,
        datasets: list[str]
    ) -> None:
        super().__init__(
            transforms,
            datasets
        )

    def _load_data(
        self
    ) -> None:
        for split in ["train", "test"]:
            root_paths = [os.path.join("data", dataset, split, "labels") for dataset in self._datasets]

            full_paths = []
            for path in root_paths:
                full_paths += [os.path.join(path, fn) for fn in os.listdir(path)]

            pool = multiprocessing.Pool(processes = os.cpu_count())
            labels = pool.map(reader.read_label, full_paths)

            pool.close()
            pool.join()

            self.data[split] = labels