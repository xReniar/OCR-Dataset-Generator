from .dataloader import Dataloader
from ..utils import reader
import multiprocessing
import os

class RecDataloader(Dataloader):
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

            # get det data
            pool = multiprocessing.Pool(processes = self._workers)
            data = pool.map(reader.read_label, full_paths)
            pool.close()
            pool.join()

            # convert it to recog data
            args = [(img_path, text_box_list) for img_path, text_box_list in data]

            rec_pool = multiprocessing.Pool(processes = self._workers)
            rec_labels = rec_pool.starmap(self.__extract_rec_info__, args)
            rec_pool.close()
            rec_pool.join()

            self.data[split] = [item for sublist in rec_labels for item in sublist]

    @staticmethod
    def __extract_rec_info__(
        filepath: str,
        text_box_list: list[tuple[str, list[int]]]
    ) -> list[tuple[str, str, list[int], str]]:
        _, img_name = os.path.split(filepath)

        labels = []
        for i, (text, box) in enumerate(text_box_list):
            crop_name = img_name.replace(".", f"-{i}.")
            labels.append((crop_name,text,box,filepath))

        return labels
    
    def _filter(self):
        pass