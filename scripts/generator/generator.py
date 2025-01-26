from abc import ABC, abstractmethod
from multiprocessing import Process
import shutil
import os
import _io
import ast


class Generator(ABC):
    base_path = "../../output"

    def __init__(
        self,
        datasets : list,
        transforms,
    ) -> None:
        super().__init__()

        new_datasets = []
        for dataset in datasets:
            element = dataset
            if "-" in element:
                dataset_root = dataset.split("-")[0]
                element = f"{dataset_root}/{element}"
            new_datasets.append(element)

        self.transforms = transforms
        self.datasets:list[str] = new_datasets

    def name(
        self
    ) -> str:
        return self.__class__.__name__.lower().replace("generator","")
    
    def read_rows(
        self,
        label_file: _io.TextIOWrapper
    ) -> list:
        labels = []
        for row in label_file.readlines():
            text, bbox = row.split("\t")
            labels.append((text, tuple(ast.literal_eval(bbox))))
        return labels
    
    def __copy_file(self, source, destination):
        shutil.copy(source, destination)
    
    def copy_file(self, label_dir, imgs_dir, current_path, split):
        args = []                
        # image creation
        for label in label_dir:
            img_name = label.replace("txt","")
            extension = [f for f in imgs_dir if f.startswith(img_name)][0].split(".")[1]

            img_name = f"{img_name}{extension}"
            src = f"{current_path}/images/{img_name}" 
            dst = f"../../output/doctr-det/{split}/images/{img_name}"
            args.append((src, dst))
        
        processes = []
        for src, dst in args:
            process = Process(target=self.__copy_file, args=(src, dst))
            processes.append(process)
            process.start()

        # Aspetta la fine di tutti i processi
        for process in processes:
            process.join()

    @abstractmethod
    def generate_det_data(
        self
    ) -> None:
        root_path = f"{self.base_path}/{self.name()}-det"
        os.makedirs(
            root_path,
            exist_ok=True
        )
        for split in ["train","test"]:
            os.makedirs(os.path.join(root_path, split), exist_ok=True)
    
    @abstractmethod
    def generate_rec_data(
        self
    ) -> None:
        os.makedirs(
            os.path.join(self.base_path, self.name() + "-rec"),
            exist_ok=True
        )