import os
import json


class Dataset():
    def __init__(
        self,
        local_datasets: list,
        online_datasets: list
    ) -> None:
        self.local_datasets: list = local_datasets
        self.online_datasets: list = online_datasets

        self.download_annotations()

    def get_path(self, dataset: str | None = None) -> str:
        datasets = self.local_datasets + self.online_datasets
        if (dataset == None and len(datasets) != 0):
            raise Exception(f"{self.get_class_name().upper()} dataset have more variants but none of them were specified")
        if (dataset != None and len(datasets) == 0):
            raise Exception(f"{self.get_class_name().upper()} dataset does not have {dataset} variant")

        base_path = os.path.join("../../data", self.get_class_name())
        if dataset != None:
            annotations = os.path.join(base_path, dataset)
        else:
            annotations = os.path.join(base_path)
        return annotations

    def download_annotations(self) -> None:
        '''
        Download all the annotations specified in `datasets` if not `None`. 
        '''

    def check_images(self, dataset: str = None) -> bool:
        '''
        Check if all the images specified in the `train` and `test` are available

        Args:
            dataset (str): The name of the dataset
        Returns:
            bool: True if all the images are in the directory, False otherwise
        '''
        datasets = self.local_datasets + self.online_datasets
        if (dataset != None and len(datasets) == 0):
            raise Exception(f"no {dataset} exists for {self.get_class_name()}")
        
        train_path = self.get_path(dataset) + f"/train"
        test_path = self.get_path(dataset) + f"/test"
        img_folder_path = self.get_path(dataset) + f"/images"
        img_folder = os.listdir(img_folder_path)

        train_check: bool = False
        for annotation in os.listdir(train_path):
            img_name = annotation.split(".")[0]
            img_exists = any(fname.startswith(img_name) for fname in img_folder)
            train_check = train_check and img_exists
            if not(train_check):
                break

        test_check: bool = False
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
        datasets = self.local_datasets + self.online_datasets

        assert (split in ["train", "test"]), f"split should be equal to 'train' or 'test', but equal to '{split}'"
        if (dataset != None and len(datasets) == 0):
            raise Exception(f"no {dataset} exists for {self.get_class_name()}")

        annotations = self.get_path(dataset) + f"/{split}"

        return os.path.isdir(annotations) and bool(os.listdir(annotations))


    def get_class_name(self) -> str:
        return self.__class__.__name__.lower()