from abc import ABC, abstractmethod
from datasets import load_dataset
import os


class Dataset(ABC):
    def __init__(
        self,
        local_datasets: list,
        online_datasets: dict,
    ) -> None:
        self.local_datasets: list = local_datasets
        self.online_datasets: dict = online_datasets

    def get_path(self, dataset: str | None = None) -> str:
        '''
        Get the path of the dataset

        Args:
            dataset (str): The name of the dataset
        Returns:
            Path of the dataset
        '''
        self.__check_parameters(dataset)

        base_path = os.path.join("../../data", self.get_class_name())
        if dataset != None:
            annotations = os.path.join(base_path, dataset)
        else:
            annotations = os.path.join(base_path)
        return annotations

    def check_images(self, dataset: str | None = None) -> bool:
        '''
        Check if all the images specified in the `train` and `test` are available

        Args:
            dataset (str): The name of the dataset
        Returns:
            bool: True if all the images are in the directory, False otherwise
        '''
        self.__check_parameters(dataset)
        
        train_path = self.get_path(dataset) + f"/train"
        test_path = self.get_path(dataset) + f"/test"
        img_folder_path = self.get_path(dataset) + f"/images"
        img_folder = os.listdir(img_folder_path)

        train_check: bool = True
        for annotation in os.listdir(train_path):
            img_name = annotation.split(".")[0]
            img_exists = any(fname.startswith(img_name) for fname in img_folder)
            train_check = train_check and img_exists
            if not(train_check):
                break

        test_check: bool = True
        for annotation in os.listdir(test_path):
            img_name = annotation.split(".")[0]
            img_exists = any(fname.startswith(img_name) for fname in img_folder)
            test_check = test_check and img_exists
            if not(test_check):
                break

        return train_check and test_check

    def check_annotations(self, split: str, dataset: str | None = None) -> bool:
        '''
        Check if the annotations folder for the given `dataset` and `split` exists.
        Also checks if the folder is not empty

        Args:
            dataset (str): The name of the dataset.
            split (str): The name of the split (e.g., "train", "test").

        Returns:
            bool: True if the annotations folder exists, False otherwise.
        '''
        assert (split in ["train", "test"]), f"split should be equal to 'train' or 'test', but equal to '{split}'"
        self.__check_parameters(dataset)

        annotations = self.get_path(dataset) + f"/{split}"

        return os.path.isdir(annotations) and bool(os.listdir(annotations))

    def get_class_name(self) -> str:
        return self.__class__.__name__.lower()
    
    def __check_parameters(self, dataset:str | None = None):
        datasets = self.local_datasets + list(self.online_datasets.keys())
        if self.get_class_name() in datasets:
            datasets.remove(self.get_class_name())
        if (dataset == None and len(datasets) != 0):
            raise Exception(f"{self.get_class_name().upper()} dataset have more variants but none of them were specified")
        if (dataset != None and len(datasets) == 0):
            raise Exception(f"{self.get_class_name().upper()} wrong parameter passed")
        if dataset != None and dataset not in datasets:
            raise Exception(f"{self.get_class_name().upper()} dataset does not have '{dataset}' variant")

    @abstractmethod
    def download_data(self, dataset: str | None = None) -> None:
        '''
        Download images and annotations for `dataset`

        Args:
            dataset (str): The name of the dataset
        '''
        self.__check_parameters(dataset)

        if dataset in self.online_datasets.keys():
            self._current_link:str = self.online_datasets[dataset if dataset != None else self.get_class_name()]

        os.makedirs(self.get_path(dataset), exist_ok=True)
        os.makedirs(f"{self.get_path(dataset)}/train", exist_ok=True)
        os.makedirs(f"{self.get_path(dataset)}/test", exist_ok=True)
        os.makedirs(f"{self.get_path(dataset)}/images", exist_ok=True)