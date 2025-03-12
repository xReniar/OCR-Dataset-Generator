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

    def _load_data(self) -> None:

        '''
        for split in ["train", "test"]:
            all_tasks = []
            for dataset in self._datasets:
                all_tasks.append((
                    dataset,
                    split
                ))
            pool = multiprocessing.Pool(processes=4)
            
            # image_path -> list[str, list[int]]
            curr_list = pool.starmap(self.__loader__, all_tasks)
            pool.close()
            pool.join()
            
            self.data[split] = curr_list
        '''

    def __loader__(
        self,
        dataset: str,
        split: str
    ) -> None:
        split_folder_path = os.path.join("data", dataset, split, "images")
        content = reader.read_labels(split_folder_path)
        curr_list = []
        for label_name in content.keys():
            img_name = label_name.strip(".txt")
            curr_list.append((
                os.path.join(split_folder_path, img_name),
                content[label_name]
            ))

        return curr_list