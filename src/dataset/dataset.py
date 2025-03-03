from abc import ABC, abstractmethod
from PIL import Image, ImageDraw
import os
import ast


class Dataset(ABC):
    '''
    self.config: configuration file
    self.sub_datasets: list of sub-datasets if it's a type 2 dataset
    self._current: specific name of the dataset
    '''
    def __init__(
        self,
        config:dict
    ) -> None:
        super().__init__()
        self.config:dict = config

        # to manage multiple sub datasets
        self.sub_datasets:list = []
        if not(len(self.config.keys()) == 1 and self._root_name() in self.config.keys() or len(self.config.keys()) == 0):
            self.sub_datasets = list(self.config.keys())
            self._current = self.sub_datasets[0]
        else:
            self._current = self._root_name()

    def _root_name(self) -> str:
        '''
        Returns root name of the dataset
        '''
        return self.__class__.__name__.lower()
    
    def set_subdataset(
        self,
        sub_dataset:str
    ) -> None:
        '''
        Set the current dataset to `sub_dataset` if the dataset has sub-datasets
        If not raises a value error
        '''
        if sub_dataset in self.config.keys():
            self._current = sub_dataset
        else:
            raise ValueError(f"Dataset {self._root_name().upper()} does not have {sub_dataset} sub-dataset")

    def path(self) -> str:
        '''
        Return the path of the directory where images and labels are stored
        '''
        base_path = os.path.join("data", self._root_name())
        if len(self.sub_datasets) != 0:
            base_path = os.path.join(base_path, self._current)

        return base_path
    
    def is_downloaded(
        self
    ) -> bool:
        return os.path.exists(self.path())
    
    def check_images(
        self
    ) -> None:
        img_dir = list(map(lambda img: img.split(".")[0], sorted(os.listdir(f"{self.path()}/images"))))
        train_dir = list(map(lambda img: img.split(".")[0], sorted(os.listdir(f"{self.path()}/train"))))
        test_dir = list(map(lambda img: img.split(".")[0], sorted(os.listdir(f"{self.path()}/test"))))

        label_dir = train_dir + test_dir
        for label in label_dir:
            if label not in img_dir:
                raise Exception(f"Missing image for {label}.txt")
            break
    
    def check_labels(
        self,
        errors: dict
    ) -> None:
        for split in ["train", "test"]:
            label_dir = sorted(os.listdir(f"{self.path()}/{split}"))
        
            for label in label_dir:
                with open(os.path.join(self.path(), split, label), "r") as file:
                    for i, row in enumerate(file.readlines()):
                        text, bbox = row.split("\t")
                        x1, y1, x2, y2 = tuple(ast.literal_eval(bbox))
                        if not(x1 < x2 and y1 < y2):
                            errors[f"{self._current}/{split}/{label}"] = dict(
                                line = i + 1,
                                text = text,
                                bbox = [x1, y1, x2, y2]
                            )

    def adjust_label_name(self):
        if len(self.config.keys()) > 0:
            path = self.path()
        
            ext_dict = {}
            img_dir = sorted(os.listdir(f"{path}/images"))
            for img_fn in img_dir:
                fn, ext = tuple(img_fn.split("."))
                ext_dict[fn] = ext

            for split in ["train", "test"]:
                for label_fn in sorted(os.listdir(f"{path}/{split}")):
                    fn, _ = tuple(label_fn.split("."))
                    os.rename(
                        f"{path}/{split}/{label_fn}",
                        f"{path}/{split}/{fn}.{ext_dict[fn]}.txt"
                    )

    def draw_labels(self,
        outline: str = "black",
        fill: str | None = None,
        width: int = 1
    ) -> None:
        os.makedirs(os.path.join(self.path(),"draw"), exist_ok=True)
        imgs_dir = os.listdir(f"{self.path()}/images")

        print(f"Draw labels for {self._current}")

        for split in ["train", "test"]:
            os.makedirs(os.path.join(self.path(),"draw", split), exist_ok=True)
            for file in os.listdir(os.path.join(self.path(),split)):
                file_path = os.path.join(self.path(), split, file)

                bbox_list = []
                with open(file_path, "r") as label:
                    for row in label.readlines():
                        _, bbox = row.split("\t")
                        bbox_list.append(ast.literal_eval(bbox))
                
                img_name = file_path.split("/")[-1].replace(".txt","")

                img = Image.open(f"{self.path()}/images/{img_name}")
                draw = ImageDraw.Draw(img)
                for bbox in bbox_list:
                    draw.rectangle(bbox, fill, outline, width)
                img.save(f"{self.path()}/draw/{split}/{img_name}")

    @abstractmethod
    def download(
        self
    ) -> None:
        '''
        Download images and labels for `current` dataset
        '''

        path = self.path()

        os.makedirs(path, exist_ok=True)
        os.makedirs(f"{path}/train", exist_ok=True)
        os.makedirs(f"{path}/test", exist_ok=True)
        os.makedirs(f"{path}/images", exist_ok=True)