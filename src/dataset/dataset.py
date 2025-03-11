from abc import ABC, abstractmethod
from ..utils import utils
import cv2
import os
import ast


class Dataset(ABC):
    def __init__(
        self,
        config:dict
    ) -> None:
        super().__init__()
        self.config: dict = config

        # to manage multiple sub datasets
        self.sub_datasets:list = []
        if not(len(self.config.keys()) == 1 and self._root_name() in self.config.keys() or len(self.config.keys()) == 0):
            self.sub_datasets = list(self.config.keys())
            self._current = self.sub_datasets[0] # default value
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
        self,
        errors: dict
    ) -> None:
        for split in ["train", "test"]:
            images_dir = set(map(lambda fn: fn.split(".")[0], sorted(os.listdir(os.path.join(self.path(), split, "images")))))
            labels_dir = set(map(lambda fn: fn.split(".")[0], sorted(os.listdir(os.path.join(self.path(), split, "labels")))))

            missing_labels = images_dir - labels_dir
            if not(missing_labels):
                errors["missing_labels"] = list(missing_labels)

            missing_images = labels_dir - images_dir
            if not(missing_images):
                errors["missing_images"] = list(missing_images)
    
    def check_labels(
        self,
        errors: dict
    ) -> None:
        for split in ["train", "test"]:
            label_dir = sorted(os.listdir(os.path.join(self.path(), split, "labels")))

            labels_content = utils.read_labels(label_dir)
            for label_filename in labels_content.keys():
                for i, (text, bbox) in enumerate(labels_content[label_filename]):
                    x1, y1, x2, y2 = tuple(ast.literal_eval(bbox))
                    if not(x1 < x2 and y1 < y2):
                        errors[f"{self._current}/{split}/labels/{label_filename}"] = dict(
                            line = i + 1,
                            text = text,
                            bbox = [x1, y1, x2, y2]
                        )

    def draw_labels(self,
        color: tuple[int] = (0, 0, 0),
        thickness: int = 1
    ) -> None:
        os.makedirs(os.path.join(self.path(),"draw"), exist_ok=True)

        print(f"Draw labels for {self._current}")

        for split in ["train", "test"]:
            label_dir = os.path.join(self.path(), split, "labels")
            img_dir = os.path.join(self.path(), split, "images")
            output_dir = os.path.join(self.path(), "draw", split)
            os.makedirs(output_dir, exist_ok=True)

            labels_content = utils.read_labels(label_dir)
            for label_filename in labels_content.keys():
                img_name = label_filename.strip(".txt")
                img = cv2.imread(os.path.join(img_dir, img_name))
                for (_, bbox) in labels_content[label_filename]:
                    img = cv2.rectangle(
                        img = img,
                        pt1 = (bbox[0], bbox[1]),
                        pt2 = (bbox[2], bbox[3]),
                        color = color,
                        thickness = thickness
                    )
                cv2.imwrite(os.path.join(output_dir, img_name), img)

    def download(self) -> None:
        # create folders
        path = self.path()
        os.makedirs(path, exist_ok=True)
        for split in ["train", "test"]:
            os.makedirs(f"{path}/{split}", exist_ok=True)
            for folder in ["images", "labels"]:
                os.makedirs(f"{path}/{split}/{folder}", exist_ok=True)

        self._download()
        self._adjust_label_name()

    def _adjust_label_name(self):
        if len(self.config.keys()) > 0:
            path = self.path()
        
            ext_dict = {}
            for split in ["train", "test"]:
                folder_path = os.path.join(path, split)
                img_dir = sorted(os.listdir(os.path.join(folder_path, "images")))
                for img_fn in img_dir:
                    fn, ext = tuple(img_fn.split("."))
                    ext_dict[fn] = ext

                for label_fn in sorted(os.listdir(os.path.join(folder_path, "labels"))):
                    fn, _ = tuple(label_fn.split("."))
                    os.rename(
                        os.path.join(folder_path, "labels", label_fn),
                        os.path.join(folder_path, "labels", f"{fn}.{ext_dict[fn]}.txt")
                    )

    @abstractmethod
    def _download(self) -> None:
        pass