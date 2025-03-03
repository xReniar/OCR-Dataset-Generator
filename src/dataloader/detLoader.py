from .dataloader import Dataloader
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

    def load_data(self, split:str) -> None:
        for dataset in self.datasets:
            img_dir = sorted(os.listdir(f"./data/{dataset}/images"))
            split_dir = sorted(os.listdir(f"./data/{dataset}/{split}"))