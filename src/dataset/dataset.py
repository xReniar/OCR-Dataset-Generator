from abc import ABC, abstractmethod
import progressbar
import threading
import time
import os


class Dataset(ABC):
    def __init__(
        self,
        config:dict
    ) -> None:
        super().__init__()
        self.config: dict = config

        # to manage multiple sub datasets
        self.sub_datasets:list = []
        if not(len(self.config.keys()) == 1 and self._root_name() in self.config.keys() or len(self.config.keys()) == 0):
            self.sub_datasets = list(self.config.keys())
            self._current = self.sub_datasets[0] # default value
        else:
            self._current = self._root_name()

    def __str__(self) -> str:
        return self._current

    def _root_name(self) -> str:
        '''
        Returns root name of the dataset
        '''
        return self.__class__.__name__.lower()
    
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
    
    def is_downloaded(
        self
    ) -> bool:
        return os.path.exists(self.path())

    def download(
        self
    ) -> None:
        if not(self.is_downloaded()):
            # create folders
            path = self.path()
            os.makedirs(path, exist_ok=True)
            for split in ["train", "test"]:
                split_folder_path = os.path.join(path, split)
                os.makedirs(split_folder_path, exist_ok=True)
                for folder in ["images", "labels"]:
                    os.makedirs(os.path.join(split_folder_path, folder), exist_ok=True)

            # download step
            downloading = True
            def progress_bar() -> None:
                widgets = ["  [", progressbar.AnimatedMarker(), f"] Downloading {self.__str__()} dataset"]
                bar = progressbar.ProgressBar(widgets=widgets, maxval=progressbar.UnknownLength).start()
                i = 0
                while downloading:
                    i += 1
                    bar.update(i)
                    time.sleep(0.1)
                bar.widgets = [f"  [✓] Downloaded {self.__str__()} dataset"]
                bar.finish()
            progress_thread = threading.Thread(target=progress_bar)
            progress_thread.start()

            self._download()
            downloading = False
            progress_thread.join()

            # adjust label name
            self._adjust_label_name()
        else:
            if len(self.config.keys()) > 0:
                print(f"  [—] {self.__str__()} dataset already downloaded")
            else:
                print(f"  [—] {self.__str__()} dataset ready")

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
    def _download(self) -> None:
        pass