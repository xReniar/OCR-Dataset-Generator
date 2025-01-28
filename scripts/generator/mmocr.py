from .generator import Generator
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

    def generate_det_data(self):
        img_folder_name = "textdet_imgs"

        for split in ["train","test"]:
            img_folder_path = f"output/{self.test_name}/{img_folder_name}/{split}"
            os.makedirs(img_folder_path,exist_ok=True)

            data_list = []
            for dataset in self.datasets:
                current_path = f"data/{dataset}"

                imgs_dir = sorted(os.listdir(f"{current_path}/images"))
                labels_dir = sorted(os.listdir(f"{current_path}/{split}"))
                extension_map = self.extension_map(imgs_dir)
                
                dst_path = f"output/{self.test_name}/{img_folder_name}/{split}"
                self.copy_file(
                    labels_dir,
                    extension_map,
                    current_path,
                    dst_path
                )

                current_data_list = []
                for label in labels_dir:
                    img = Image.open(f"{current_path}/images/{extension_map[label]}")
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
                        img_path = f"{img_folder_name}/{split}/{extension_map[label]}",
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

            with open(f"output/{self.test_name}/textdet_{split}.json", "w") as file:
                json.dump(label, file, indent=4)



    def generate_rec_data(self):
        img_folder_name = "textrecog_imgs"

        for split in ["train","test"]:
            img_folder_path = f"output/{self.test_name}/{img_folder_name}/{split}"
            os.makedirs(img_folder_path,exist_ok=True)

            data_list = []
            for dataset in self.datasets:
                current_path = f"data/{dataset}"

                imgs_dir = sorted(os.listdir(f"{current_path}/images"))
                labels_dir = sorted(os.listdir(f"{current_path}/{split}"))
                extension_map = self.extension_map(imgs_dir)

                for label in labels_dir:
                    img = Image.open(f"{current_path}/images/{extension_map[label]}")
                    for index, (text, bbox) in enumerate(self.read_rows(f"{current_path}/{split}/{label}")):
                        crop_name = extension_map[label].replace(".",f"-{index}.")
                        img.crop(bbox).save(f"output/{self.test_name}/{img_folder_name}/{split}/{crop_name}")
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
            with open(f"output/{self.test_name}/textrecog_{split}.json", "w") as file:
                json.dump(label, file, indent=4)