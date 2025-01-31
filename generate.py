from scripts.dataset import *
from scripts.generator import *
import json


def generate(
    test_name: str,
    ocr_system: str,
    tasks: list,
    datasets: dict
) -> None:
    
    # create dataset objects
    for dataset in datasets.keys():
        if "-" in dataset:
            root:str = dataset.split("-")[0].upper()
            dataset_instance:Dataset = DATASETS[root](CONFIG[root])
            dataset_instance.set_to(dataset)
            datasets[dataset] = dataset_instance
        else:
            datasets[dataset] = DATASETS[dataset.upper()](CONFIG[dataset.upper()])

    # download if necessary
    for dataset in datasets.keys():
        dataset_instance: Dataset = datasets[dataset]
        if not(dataset_instance.is_downloaded()):
            dataset_instance.download()

    # check if all the labels have a corresponding image
    for dataset in datasets.keys():
        dataset_instance: Dataset = datasets[dataset]
        dataset_instance.check()

    ocr_generator: Generator = OCR_SYSTEMS[ocr_system](test_name,list(datasets.keys()))

    for task in tasks:
        if task == "det":
            print("Start Generating data for text detection")
            ocr_generator.generate_det_data()
        if task == "rec":
            print("Start Generating data for text recognition")
            ocr_generator.generate_rec_data()
    

if __name__ == "__main__":
    config:dict = json.load(open("config/config.json", "r"))
    datasets:dict = config["datasets"]

    defined_classes = list(map(lambda x: str(x).lower(), list(DATASETS.keys())))
    config_classes = list(datasets.keys())
    if len(defined_classes) > len(config_classes):
        raise BaseException("There are some datasets that are not in the `config.json`")
    if len(defined_classes) < len(config_classes):
        raise BaseException("Some datasets in the `config.json` do not have a script in `./scripts/dataset/`")

    selected_datasets = {}
    for dataset in list(datasets.keys()):
        value = datasets[dataset]
        if isinstance(value, str):
            if value == "y":
                selected_datasets[dataset] = None
        elif isinstance(value, dict):
            for sub in value.keys():
                if value[sub] == "y":
                    selected_datasets[sub] = None
        else:
            raise ValueError(f"Type '{type(value)}' not available")
    
    generate(
        config["test-name"],
        config["ocr-system"],
        config["tasks"],
        selected_datasets
    )