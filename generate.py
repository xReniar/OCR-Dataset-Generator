from scripts.dataset import *
from scripts.generator import *
import json
import argparse


def main(
    test_name: str,
    ocr_system: str,
    tasks: list,
    datasets: dict,
    args
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
        dataset_instance: Dataset = datasets[dataset]
        if not(dataset_instance.is_downloaded()):
            dataset_instance.download()

    # check if all the labels have a corresponding image
    # also check if there are wrong bounding boxes
    errors = {}
    for dataset in datasets.keys():
        dataset_instance: Dataset = datasets[dataset]
        dataset_instance.check_images()
        dataset_instance.check_labels(errors)

    if args.draw:
        for dataset in datasets.keys():
            dataset_instance: Dataset = datasets[dataset]
            dataset_instance.draw_labels()
    if args.generate:
        if not(len(list(errors.keys())) != 0):
            ocr_generator: Generator = OCR_SYSTEMS[ocr_system](test_name,list(datasets.keys()))

            for task in tasks:
                ocr_generator.generate_data(task)
        else:
            with open("errors.json", "w") as error_file:
                json.dump(errors, error_file, indent=4, ensure_ascii=False)
            print("Some bbox values are wrong, details in `./errors.json`. " + 
                "Correct them before generating data")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training data generator for Text Detection and Text Recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--draw", action="store_true", help="Avvia la modalità di disegno")
    group.add_argument("--generate", action="store_true", help="Avvia la modalità di generazione")

    args = parser.parse_args()

    return args

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
    
    args = parse_args()

    main(
        config["test-name"],
        config["ocr-system"],
        config["tasks"],
        selected_datasets,
        args
    )