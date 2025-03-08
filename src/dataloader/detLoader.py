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
                    img_fn = label_fn.replace(".txt","")
                    curr_list.append({
                        f"{img_fn}": list(map(lambda x: x[1],self.read_label(f"{split_folder_path}/{label_fn}")))
                    })
                self.data[dataset][split] = curr_list