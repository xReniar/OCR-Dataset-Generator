from .generator import Generator
from ..dataloader import Dataloader


class EasyOCRGenerator(Generator):
    def __init__(
        self,
        test_name: str,
        datasets: list, 
        lang: list[str] | None,
        workers: int,
        transforms = None
    ) -> None:
        super().__init__(
            test_name,
            datasets,
            lang,
            workers,
            transforms
        )

    def _generate(
        self,
        dataloader: Dataloader,
        task: str,
        process
    ) -> None:
        pass

    def _det(
        self,
        img_output_path: str,
        img_path: str,
        gt: list
    ) -> None:
        pass

    def _rec(
        self,
        img_output_path: str,
        img_path: str,
        gt: list
    ) -> None:
        pass