from abc import ABC, abstractmethod
import os


class Dataset(ABC):
    def __init__(
        self,
        config:dict
    ) -> None:
        super().__init__()
        self.config: dict = config
        self.sub_datasets:list = []
        self._current = ""

        if len(self.config.keys()) == 1:
            if self._root_name() in self.config.keys():
                self._current = self._root_name()
            elif "__name__" in self.config.keys():
                self._current = self.config["__name__"]
        elif len(self.config.keys()) > 1:
            self.sub_datasets = list(self.config.keys())
            self._current = self.sub_datasets[0] # default value

    def __str__(self) -> str:
        return self._current
    
    def set_subdataset(
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
            raise ValueError(f"Dataset {self._root_name().upper()} does not have {sub_dataset} sub-dataset")

    def path(
        self
    ) -> str:
        '''
        Return the path of the directory where images and labels are stored
        '''
        base_path = os.path.join("data", self._root_name())
        if len(self.sub_datasets) != 0:
            base_path = os.path.join(base_path, self.__str__())

        return base_path

    def _adjust_label_name(
        self
    ) -> None:
        if len(self.config.keys()) > 0:
            path = self.path()
        
            ext_dict = {}
            for split in ["train", "test"]:
                folder_path = os.path.join(path, split)
                img_dir = sorted(os.listdir(os.path.join(folder_path, "images")))
                for img_fn in img_dir:
                    fn, ext = tuple(img_fn.split("."))
                    ext_dict[fn] = ext

                for label_fn in sorted(os.listdir(os.path.join(folder_path, "labels"))):
                    fn, _ = tuple(label_fn.split("."))
                    os.rename(
                        os.path.join(folder_path, "labels", label_fn),
                        os.path.join(folder_path, "labels", f"{fn}.{ext_dict[fn]}.txt")
                    )
    
    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def _root_name(self) -> str:
        pass