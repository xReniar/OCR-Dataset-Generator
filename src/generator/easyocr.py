from .generator import Generator
from ..dataloader import Dataloader


class EasyOCRGenerator(Generator):
    def __init__(
        self,
        test_name: str,
        datasets: list, 
        dict: list[str] | None,
        workers: int,
        augmentation: bool
    ) -> None:
        super().__init__(
            test_name,
            datasets,
            dict,
            workers,
            augmentation
        )

    def _generate(
        self,
        dataloader: Dataloader,
        task: str
    ) -> None:
        pass

    def _det(
        self,
        img_output_path: str,
        img_path: str,
        gt: list,
        transform: tuple[str, callable] = (None, None)
    ) -> None:
        pass

    def _rec(
        self,
        img_output_path: str,
        img_path: str,
        gt: list,
        transform: tuple[str, callable] = (None, None)
    ) -> None:
        pass