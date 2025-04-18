import albumentations as A


class DataAugmenter():
    def __init__(
        self
    ) -> None:
        self.transforms = None

    def get_operations(self) -> list:
        self.transforms = {
            "blur": A.Blur(blur_limit=7),
            "sharpen": A.Sharpen(alpha=0.5),
            "brightness": A.RandomBrightnessContrast(brightness_limit=0.2),
        }