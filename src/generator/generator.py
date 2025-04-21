from abc import ABC, abstractmethod
from ..dataloader import Dataloader
from ..augmenter import DataAugmenter
from yaspin import yaspin
from yaspin.spinners import Spinners
from colorama import Fore
import multiprocessing
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

        print(f"{Fore.LIGHTCYAN_EX}[Dataloader creation]{Fore.RESET}")
        with yaspin(text=f"Creating dataloader", spinner=Spinners.line) as spinner:
            dataloader = Dataloader(
                datasets = self.datasets,
                dict = self.dict
            )
            spinner.text = "Dataloader created"
            spinner.ok(f"{Fore.GREEN}✓{Fore.RESET}")
        print(f"- {len(dataloader.data['train'])} train images")
        print(f"- {len(dataloader.data['test'])} test images\n")

        print(f"{Fore.LIGHTCYAN_EX}[Training data generation for {self.name()}]{Fore.RESET}")
        for task_name in tasks.keys():
            if tasks[task_name] == "y":
                with yaspin(text=f"Generating {task_name} data", spinner=Spinners.line) as spinner:
                    start_time = time.time()
                    if task_name == "det":
                        self._generate(dataloader, "Detection")
                    elif task_name == "rec":
                        self._generate(dataloader, "Recognition")
                    end_time = time.time()
                    
                    spinner.text = f"Generated {task_name} data in {end_time - start_time:.2f} s"
                    spinner.ok(f"{Fore.GREEN}✓{Fore.RESET}")

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
            args = [(img_output_path, img_path, gt, transform) for (img_path, gt) in dataloader.data[split]]

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