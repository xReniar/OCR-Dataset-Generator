from generator import Generator


class PaddleOCRGenerator(Generator):
    def __init__(self,
        transforms
    ) -> None:
        super().__init__(
            transforms
        )

    def generate_det_data(self):
        super().generate_det_data()

    def generate_rec_data(self):
        super().generate_rec_data()