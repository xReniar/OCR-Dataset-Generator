from .generator import Generator
from ..dataloader.detLoader import DetDataloader
from ..dataloader.recLoader import RecDataloader
from PIL import Image
import os
import json


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

    def _generate_det_data(self, dataloader: DetDataloader):
        root_path = os.path.join(self.root_path, "Detection")
        os.makedirs(root_path, exist_ok=True)

        img_folder_name = "imgs"

        for split in ["train","test"]:
            img_folder_path = f"{root_path}/{img_folder_name}/{split}"
            os.makedirs(img_folder_path,exist_ok=True)

            data_list = []
            for dataset in self.datasets:
                current_path = f"data/{dataset}"

                labels_dir = sorted(os.listdir(f"{current_path}/{split}"))
                
                dst_path = f"{root_path}/{img_folder_name}/{split}"
                self.copy_file(
                    labels_dir,
                    current_path,
                    dst_path
                )

                current_data_list = []
                for label in labels_dir:
                    img_fn = label.replace(".txt","")
                    img = Image.open(f"{current_path}/images/{img_fn}")
                    instances = []
                    for (_, bbox) in self.read_rows(f"{current_path}/{split}/{label}"):
                        x1, y1, x2, y2 = list(map(lambda point: float(point), bbox))
                        instances.append(dict(
                            polygon = [x1, y1, x2, y1, x2, y2, x1, y2],
                            bbox = [x1, y1, x2, y2],
                            bbox_label = 0,
                            ignore = False
                        ))
                    current_data_list.append(dict(
                        instances = instances,
                        img_path = f"{img_folder_name}/{split}/{img_fn}",
                        width = img.width,
                        height = img.height
                    ))
                data_list.append(current_data_list)

            label = dict(
                metainfo = dict(
                    dataset_type = "TextDetDataset",
                    task_name = "textdet",
                    category = [dict(id = 0,name = "text")]
                ),
                data_list = data_list
            )

            with open(f"{root_path}/{split}.json", "w") as file:
                json.dump(label, file, indent=4)

    def _generate_rec_data(self, dataloader: RecDataloader):
        root_path = os.path.join(self.root_path, "Recognition")
        os.makedirs(root_path, exist_ok=True)
        img_folder_name = "imgs"

        for split in ["train","test"]:
            img_folder_path = f"{root_path}/{img_folder_name}/{split}"
            os.makedirs(img_folder_path,exist_ok=True)

            data_list = []
            for dataset in self.datasets:
                current_path = f"data/{dataset}"

                labels_dir = sorted(os.listdir(f"{current_path}/{split}"))

                for label in labels_dir:
                    img_fn = label.replace(".txt","")
                    img = Image.open(f"{current_path}/images/{img_fn}")
                    for index, (text, bbox) in enumerate(self.read_rows(f"{current_path}/{split}/{label}")):
                        crop_name = img_fn.replace(".",f"-{index}.")
                        img.crop(bbox).save(f"{root_path}/{img_folder_name}/{split}/{crop_name}")
                        data_list.append(dict(
                            instances = [{"text": text}],
                            img_path = f"{img_folder_name}/{split}/{crop_name}"
                        ))
                    img.close()

            label = dict(
                metainfo = dict(
                    dataset_type = "TextRecogDataset",
                    task_name = "textrecog"
                ),
                data_list = data_list
            )
            with open(f"{root_path}/{split}.json", "w") as file:
                json.dump(label, file, indent=4, ensure_ascii=False)