from abc import ABC, abstractmethod
from PIL import Image, ImageDraw
import os
import ast
import json


class Dataset(ABC):
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

        #self.mode = "online" if len(config.keys()) > 0 else "local"
        

    def _root_name(self) -> str:
        '''
        Returns root name of the dataset
        '''
        return self.__class__.__name__.lower()
    
    def set_to(
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



    def draw_labels(self,
        outline: str = "black",
        fill: str | None = None,
        width: int = 1
    ) -> None:
        os.makedirs(os.path.join(self.path(),"draw"), exist_ok=True)
        imgs_dir = os.listdir(f"{self.path()}/images")

        print(f"Draw labels for {self._root_name()}")

        extension_map = {}
        for img in imgs_dir:
            name, ext = img.split(".")
            extension_map[name] = ext
        for split in ["train", "test"]:
            os.makedirs(os.path.join(self.path(),"draw", split), exist_ok=True)
            for file in os.listdir(os.path.join(self.path(),split)):
                file_path = os.path.join(self.path(), split, file)

                bbox_list = []
                with open(file_path, "r") as label:
                    for row in label.readlines():
                        _, bbox = row.split("\t")
                        bbox_list.append(ast.literal_eval(bbox))
                
                img_name = file_path.split("/")[-1].replace("txt","")
                extension = extension_map[img_name.replace(".","")]
                

                img = Image.open(f"{self.path()}/images/{img_name}{extension}")
                draw = ImageDraw.Draw(img)
                for bbox in bbox_list:
                    draw.rectangle(bbox, fill, outline, width)
                img.save(f"{self.path()}/draw/{split}/{img_name}{extension}")

    def is_downloaded(
        self
    ) -> bool:
        if os.path.exists(self.path()):
            print(f"{self._current} already downloaded")
            return True
        else:
            print(f"Downloading {self._current}")
            return False
    
    def has_variants(
        self
    ) -> bool:
        keys = list(self.config.keys())
        if len(keys) == 0:
            return False
        if self._root_name() in keys:
            return False
        else:
            return True

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