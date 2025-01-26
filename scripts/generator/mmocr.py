from .generator import Generator


class MMOCRGenerator(Generator):
    def __init__(
        self,
        test_name: str,
        datasets: list,
        transforms = None
    ) -> None:
        super().__init__(
            test_name,
            datasets,
            transforms
        )

    def generate_det_data(self):
        super().generate_det_data()

    def generate_rec_data(self):
        super().generate_rec_data()