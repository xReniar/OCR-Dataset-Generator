import albumentations as A


class DataAugmenter():
    def __init__(
        self,
        config: dict,
    ) -> None:
        pass

    def augment(self) -> list:
        return [
            A.Compose([
                A.RandomCrop(width=512, height=512),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=90, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(p=0.5),
                A.Blur(blur_limit=(3, 7), p=0.5),
                A.HueSaturationValue(p=0.5),
            ]),
            A.Compose([
                A.RandomCrop(width=512, height=512),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=90, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.GaussNoise(p=0.5),
                A.Blur(blur_limit=(3, 7), p=0.5),
                A.HueSaturationValue(p=0.5),
            ])
        ]