from .dataloader import Dataloader
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

    def _load_data(self) -> None:
        for dataset in sorted(self.datasets):
            self.data[dataset] = dict(
                train=[],
                test=[]
            )
            for split in ["train", "test"]:
                curr_list = []
                split_folder_path = f"./data/{dataset}/{split}"
                for label_fn in sorted(os.listdir(split_folder_path)):
                    img_fn, img_type, _ = tuple(label_fn.split("."))
                    for i, (text, bbox) in enumerate(self.read_label(f"{split_folder_path}/{label_fn}")): 
                        curr_list.append({
                            f"{img_fn}-{i}.{img_type}": [bbox,text]
                        })
                self.data[dataset][split] = curr_list