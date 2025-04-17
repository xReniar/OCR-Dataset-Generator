from abc import ABC, abstractmethod
from ..dataloader import Dataloader
import progressbar
import threading
import time
import os


class Generator(ABC):
    base_path = "output"

    def __init__(
        self,
        test_name: str,
        datasets : list[str],
        dict: list[str] | None,
        workers: int,
        transforms
    ) -> None:
        super().__init__()

        new_datasets = []
        for dataset in datasets:
            element = dataset
            if "-" in element:
                dataset_root = dataset.split("-")[0]
                element = os.path.join(dataset_root, element)
            new_datasets.append(element)

        self.test_name = test_name
        self.datasets:list[str] = new_datasets
        self.dict = dict
        self.workers = workers
        self.transforms = transforms

    def name(
        self
    ) -> str:
        return self.__class__.__name__.lower().replace("generator","")

    def generate_data(self, tasks:dict):
        self.root_path = os.path.join(self.base_path,f"{self.test_name}-{self.name()}")
        os.makedirs(self.root_path, exist_ok=True)

        print("\nCreating dataloader")
        dataloader = Dataloader(
            datasets = self.datasets,
            dict = self.dict
        )
        print("Dataloader created\n")
        print(f"Generating training data for {self.name()}")

        generating = None
        def progress_bar(task: str):
            widgets = ["  [", progressbar.AnimatedMarker(), f"] Generating {task} data"]
            bar = progressbar.ProgressBar(widgets=widgets, maxval=progressbar.UnknownLength).start()
            i = 0
            while generating:
                i += 1
                bar.update(i)
                time.sleep(0.1)
            bar.widgets =  [f"  [✓] Generated {task} data"]
            bar.finish()

        for task_name in tasks.keys():
            generating = True
            if tasks[task_name] == "y":
                progress_thread = threading.Thread(target=progress_bar, args=(task_name,))
                progress_thread.start()
                if task_name == "det":
                    self._generate(dataloader, "Detection", self._det)
                elif task_name == "rec":
                    self._generate(dataloader, "Recognition", self._rec)
                generating = False
                progress_thread.join()


    @abstractmethod
    def _generate(self, dataloader: Dataloader, task: str, process) -> None:
        pass

    @abstractmethod
    def _det(self, img_output_path: str, img_path: str, gt: list) -> None:
        pass
    
    @abstractmethod
    def _rec(self, img_output_path: str, img_path: str, gt: list) -> None:
        pass