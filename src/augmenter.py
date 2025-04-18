import albumentations as A


class DataAugmenter():
    def __init__(
        self
    ) -> None:
        self.transforms = None

    def get_operations(self) -> list:
        self.transforms = {
            "blur": A.Blur(
                blur_limit=7,
                p=1.0
            )
        }

        return list(self.transforms.items())