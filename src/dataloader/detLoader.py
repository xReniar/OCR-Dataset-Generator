from .dataloader import Dataloader
import multiprocessing
import ast
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

            self.data[split] = self.__loader__(full_paths)

    def __loader__(
        self,
        label_dir_paths: str 
    ) -> list[tuple[str, list[tuple[str, list[int]]]]]:
        pool = multiprocessing.Pool(processes=os.cpu_count())
        labels = pool.map(self.label_info, label_dir_paths)

        pool.close()
        pool.join()

        return labels
    
    @staticmethod
    def label_info(
        label_path: str
    ) -> tuple[str, list[tuple[str, list[int]]]]:
        with open(label_path, "r") as label_file:
            label_content = []
            rows = list(map(lambda row: row.strip("\n").split("\t"), label_file.readlines()))
            for (text, bbox) in rows:
                label_content.append((text, ast.literal_eval(bbox)))
        
            return (label_path.strip(".txt").replace("labels", "images"), label_content)