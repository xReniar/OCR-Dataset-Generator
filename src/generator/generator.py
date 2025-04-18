from abc import ABC, abstractmethod
from ..dataloader import Dataloader
from ..augmenter import DataAugmenter
import multiprocessing
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
        augmentation: bool
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

        dataAugmenter = DataAugmenter()
        self.transforms = [(None, None)] + dataAugmenter.get_operations() if augmentation else [(None, None)]

    def name(
        self
    ) -> str:
        """
        Returns the name of the generator in lowercase
        """
        return self.__class__.__name__.lower().replace("generator","")

    def generate_data(
        self,
        tasks: dict
    ) -> None:
        """
        Generates the data for the given tasks.
        Args:
            tasks (dict): A dictionary with the tasks to generate.
        
        """
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
                    self._generate(dataloader, "Detection")
                elif task_name == "rec":
                    self._generate(dataloader, "Recognition")
                generating = False
                progress_thread.join()

    def run_process(
        self,
        img_output_path: str,
        dataloader: Dataloader,
        task: str,
        split: str
    ):
        process_map = { "Detection": self._det, "Recognition": self._rec }
        results = []
        for transform in self.transforms:
            operation = transform if transform else (None, None)
            args = [(img_output_path, img_path, gt, operation) for (img_path, gt) in dataloader.data[split]]

            with multiprocessing.Pool(processes=self.workers) as pool:
                results += pool.starmap(process_map[task], args)
        
        return results

    @abstractmethod
    def _generate(self, dataloader: Dataloader, task: str) -> None:
        pass

    @abstractmethod
    def _det(self, img_output_path: str, img_path: str, gt: list) -> None:
        pass
    
    @abstractmethod
    def _rec(self, img_output_path: str, img_path: str, gt: list) -> None:
        pass