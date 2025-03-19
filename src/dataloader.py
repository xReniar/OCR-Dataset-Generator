from abc import ABC
from .utils import reader
import multiprocessing
import os


class Dataloader(ABC):
    def __init__(
        self,
        datasets: list[str],
        lang: list[str] | None
    ) -> None:
        super().__init__()

        self._datasets = datasets
        self._lang = set(lang) if lang != None else lang
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
                self.data[split] = self.__filter__(pool.map(reader.read_label, full_paths))
    
    def __filter__(
        self,
        labels: list[tuple[str, list[tuple[str, list[int]]]]]
    ) -> list[tuple[str, list[tuple[str, list[int]]]]]:
        if self._lang != None:
            new_labels = []
            for (img_name, img_labels) in labels:
                filtered_labels = []
                for (text, bbox) in img_labels:
                    if all(char in self._lang for char in text):
                        filtered_labels.append((text, bbox))
                if len(filtered_labels) != 0:
                    new_labels.append((img_name, filtered_labels))
            return new_labels
        else:
            return labels
