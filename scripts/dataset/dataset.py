from abc import ABC, abstractmethod
import os

class Dataset(ABC):
    def __init__(
        self,
        config:dict
    ) -> None:
        super().__init__()
        self.config:dict = config

        # to manage multiple sub datasets
        self.sub_datasets:list = []
        if not(len(self.config.keys()) == 1 and self.__root_name() in self.config.keys() or len(self.config.keys()) == 0):
            self.sub_datasets = list(self.config.keys())
            self._current = self.sub_datasets[0]
        else:
            self._current = self.__root_name()

        #self.mode = "online" if len(config.keys()) > 0 else "local"
        

    def __root_name(self) -> str:
        '''
        Returns root name of the dataset
        '''
        return self.__class__.__name__.lower()
    
    def set_to(
        self,
        sub_dataset:str
    ) -> None:
        '''
        Set the current dataset to `sub_dataset` if the dataset has sub-datasets
        If not raises a value error
        '''
        if sub_dataset in self.config.keys():
            self._current = sub_dataset
        else:
            raise ValueError(f"Dataset {self.__root_name().upper()} does not have {sub_dataset} sub-dataset")

    def path(self) -> str:
        '''
        Return the path of the directory where images and labels are stored
        '''
        base_path = os.path.join("../../data", self.__root_name())
        if len(self.sub_datasets) != 0:
            base_path = os.path.join(base_path, self._current)

        return base_path

    def load_data(
        self,
        task: str,
        split: str
    ) -> None:
        assert split in ["train", "test"], f"No split of type: {split}"

        if task == "det":
            return self.__det_loader(split)
        if task == "rec":
            return self.__rec_loader(split)

    def __det_loader(self, split:str):
        base_path = f"{self.path()}/{split}"

        loader = []
        for file in os.listdir(base_path):
            labels = open(f"{base_path}/{file}","r")
            for label in labels.readlines():
                row = label.replace("\n","").split("\t")
                print(row)
            labels.close()

    def __rec_loader(self, split:str):
        for file in os.listdir(f"{self.path()}/{split}"):
            pass

    @abstractmethod
    def download(
        self
    ) -> None:
        '''
        Download images and labels for `current` dataset
        '''

        path = self.path()

        os.makedirs(path, exist_ok=True)
        os.makedirs(f"{path}/train", exist_ok=True)
        os.makedirs(f"{path}/test", exist_ok=True)
        os.makedirs(f"{path}/images", exist_ok=True)