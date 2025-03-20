from .generator import Generator
from ..dataloader import Dataloader


class YOLOTrOCRGenerator(Generator):
    def __init__(
        self,
        test_name: str,
        datasets: list,
        lang ,
        transforms
    ) -> None:
        super().__init__(
            test_name,
            datasets,
            lang,
            transforms
        )

    def _generate(
        self,
        dataloader: Dataloader,
        task: str,
        process,
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
    ) -> list[tuple[str, str]]:
        pass