from generator import Generator


class DoctrGenerator(Generator):
    def __init__(
        self,
        datasets:list,
        transforms = None
    ) -> None:
        super().__init__(
            datasets,
            transforms
        )

    def generate_det_data(self):
        super().generate_det_data()

        for dataset in self.datasets:
            print(dataset)

    def generate_rec_data(self):
        super().generate_rec_data()    


x = DoctrGenerator(["funsd", "xfund-de"])

x.generate_det_data()