from abc import ABC, abstractmethod
from multiprocessing import Process
from ..dataloader.detLoader import DetDataloader
from ..dataloader.recLoader import RecDataloader
import shutil
import yaml
import os
import ast


class Generator(ABC):
    base_path = "output"

    def __init__(
        self,
        test_name: str,
        datasets : list,
        transforms
    ) -> None:
        super().__init__()

        new_datasets = []
        for dataset in datasets:
            element = dataset
            if "-" in element:
                dataset_root = dataset.split("-")[0]
                element = f"{dataset_root}/{element}"
            new_datasets.append(element)

        self.test_name = test_name
        self.datasets:list[str] = new_datasets
        self.transforms = transforms

    def name(
        self
    ) -> str:
        return self.__class__.__name__.lower().replace("generator","")
    
    def read_rows(
        self,
        label_filepath: str
    ) -> list:
        labels = []
        label_file = open(label_filepath,"r")
        for row in label_file.readlines():
            text, bbox = row.split("\t")
            labels.append((text, tuple(ast.literal_eval(bbox))))
        label_file.close()
        return labels
    
    def __copy_file(self, source, destination):
        shutil.copy(source, destination)
    
    def copy_file(self, label_dir, src_path, dst_path):
        args = []                
        # image creation
        for label in label_dir:
            img_name = label.replace(".txt", "")
            
            src = f"{src_path}/images/{img_name}" 
            dst = f"{dst_path}/{img_name}"
            args.append((src, dst))
        
        processes = []
        for src, dst in args:
            process = Process(target=self.__copy_file, args=(src, dst))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

    def generate_data(self, tasks:str):
        self.root_path = f"{self.base_path}/{self.test_name}-{self.name()}"
        os.makedirs(self.root_path, exist_ok=True)

        if tasks["det"] == "y":
            detLoader = DetDataloader(self.transforms["detection"], self.datasets)
            self._generate_det_data(detLoader)
        if tasks["rec"] == "y":
            recLoader = RecDataloader(self.transforms["recognition"], self.datasets)
            self._generate_rec_data(recLoader)


    @abstractmethod
    def _generate_det_data(self, dataloader: DetDataloader) -> None:
        pass
    
    @abstractmethod
    def _generate_rec_data(self, dataloader: RecDataloader) -> None:
        pass