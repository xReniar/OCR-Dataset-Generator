from .dataset import Dataset
from abc import ABC
from colorama import Fore


class LocalDataset(Dataset, ABC):
    def __init__(
        self,
        config: dict
    ) -> None:
        super().__init__(config)
    
    def setup(self):
        #self._adjust_label_name()
        print(f"{Fore.GREEN}✓{Fore.RESET} {self.__str__()} dataset is local, no need to download")

    def _root_name(self) -> str:
        return self._current