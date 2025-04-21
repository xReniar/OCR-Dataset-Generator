from .dataset import Dataset
from abc import ABC, abstractmethod
from colorama import Fore
import progressbar
import threading
import contextlib
import time
import io
import os


class OnlineDataset(Dataset, ABC):
    def __init__(
        self,
        config:dict
    ) -> None:
        super().__init__(config)

    def setup(self):
        self.download()
    
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
                widgets = [" ", progressbar.AnimatedMarker(), f" Downloading {self.__str__()} dataset"]
                bar = progressbar.ProgressBar(widgets=widgets, maxval=progressbar.UnknownLength).start()
                i = 0
                while downloading:
                    i += 1
                    bar.update(i)
                    time.sleep(0.1)
                bar.widgets = [f"{Fore.GREEN}✓{Fore.RESET} Downloaded {self.__str__()} dataset"]
                bar.finish()
            progress_thread = threading.Thread(target=progress_bar)
            progress_thread.start()

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self._download()
            downloading = False
            progress_thread.join()

            # adjust label name
            self._adjust_label_name()
        else:
            print(f"{Fore.GREEN}✓{Fore.RESET} {self.__str__()} dataset already downloaded")

    @abstractmethod
    def _download(self) -> None:
        pass

    def _root_name(self) -> str:
        return self.__class__.__name__.lower()