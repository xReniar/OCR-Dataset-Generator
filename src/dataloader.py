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

        self.datasets = datasets
        self.dict = set(dict) if dict != None else dict
        self.data = { "train": [],"test": [] }
        self.workers = os.cpu_count()
        
        self.load_data()

    def load_data(self) -> None:
        """
        Loads the data from the datasets.
        """
        for split in ["train", "test"]:
            root_paths = [os.path.join("data", dataset, split, "labels") for dataset in self.datasets]

            full_paths = []
            for path in root_paths:
                full_paths += [os.path.join(path, fn) for fn in os.listdir(path)]

            with multiprocessing.Pool(processes=self.workers) as pool:
                self.data[split] = self.filter(pool.map(reader.read_label, full_paths))
    
    def filter(
        self,
        labels: list[tuple[str, list[tuple[str, list[int]]]]]
    ) -> list[tuple[str, list[tuple[str, list[int]]]]]:
        """
        Filters the labels based on the dictionary. If the dictionary is None, it returns the labels as is.

        Args:
            labels (list[tuple[str, list[tuple[str, list[int]]]]]): The labels to filter.
        
        Returns:
            list[tuple[str, list[tuple[str, list[int]]]]]: The filtered labels.
        """
        if self.dict != None:
            new_labels = []
            for (img_name, img_labels) in labels:
                filtered_labels = []
                for (text, bbox) in img_labels:
                    if all(char in self.dict for char in text):
                        filtered_labels.append((text, bbox))
                if len(filtered_labels) != 0:
                    new_labels.append((img_name, filtered_labels))
            return new_labels
        else:
            return labels
