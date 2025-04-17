from abc import ABC
from .utils import reader
import multiprocessing
import os


class Dataloader(ABC):
    def __init__(
        self,
        datasets: list[str],
        dict: list[str] | None
    ) -> None:
        super().__init__()

        self._datasets = datasets
        self._dict = set(dict) if dict != None else dict
        self.data = { "train": [],"test": [] }
        self._workers = os.cpu_count()
        
        self.load_data()

    def load_data(self) -> None:
        for split in ["train", "test"]:
            root_paths = [os.path.join("data", dataset, split, "labels") for dataset in self._datasets]

            full_paths = []
            for path in root_paths:
                full_paths += [os.path.join(path, fn) for fn in os.listdir(path)]

            with multiprocessing.Pool(processes=self._workers) as pool:
                self.data[split] = self.filter(pool.map(reader.read_label, full_paths))
    
    def filter(
        self,
        labels: list[tuple[str, list[tuple[str, list[int]]]]]
    ) -> list[tuple[str, list[tuple[str, list[int]]]]]:
        if self._dict != None:
            new_labels = []
            for (img_name, img_labels) in labels:
                filtered_labels = []
                for (text, bbox) in img_labels:
                    if all(char in self._dict for char in text):
                        filtered_labels.append((text, bbox))
                if len(filtered_labels) != 0:
                    new_labels.append((img_name, filtered_labels))
            return new_labels
        else:
            return labels
