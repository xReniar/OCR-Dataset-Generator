from .dataloader import Dataloader

class RecDataloader(Dataloader):
    def __init__(
        self,
        transforms: dict,
        datasets: list[str]
    ) -> None:
        super().__init__(
            transforms,
            datasets
        )

    def load_data(self, split:str) -> None:
        pass