from src.dataset import DATASETS, CONFIG
from src.generator import *
from src.utils.draw import draw_labels
from src.utils. dataset import check_images, check_labels
import argparse
import os
import json
import yaml


def pipeline(
    test_name: str,
    ocr_system: str,
    tasks: dict,
    datasets: dict,
    augmentation_config,
    args
) -> None:
    print("Downloading selected datasets")
    
    # create dataset objects
    for dataset in datasets.keys():
        if "-" in dataset:
            root:str = dataset.split("-")[0].upper()
            dataset_instance = DATASETS[root](CONFIG[root])
            dataset_instance.set_subdataset(dataset)
            datasets[dataset] = dataset_instance
        else:
            datasets[dataset] = DATASETS[dataset.upper()](CONFIG[dataset.upper()])
        
        # download if necessary
        dataset_instance = datasets[dataset]    
        dataset_instance.download()

    # check if all the labels have a corresponding image and viceversa
    # also check if there are wrong bounding boxes
    if os.path.isfile("./errors.json"):
        os.remove("./errors.json")

    print("\nChecking errors")
    errors = {}
    for dataset in datasets.keys():
        dataset_instance = datasets[dataset]
        missing = check_images(dataset_instance.path())
        missing_images = missing["missing_images"]
        missing_labels = missing["missing_labels"]

        label_errors = check_labels(dataset_instance.path())

        if not(len(missing_images) == 0 and len(missing_labels) == 0 and len(label_errors.keys()) == 0):
            errors[dataset_instance.__str__()] = dict(
                missing_images = missing["missing_images"],
                missing_labels = missing["missing_labels"],
                label_checking = label_errors
            )

    if len(list(errors.keys())) != 0:
        with open("errors.json", "w") as error_file:
            json.dump(errors, error_file, indent=4, ensure_ascii=False)
        print("  [✗] Some bbox values are wrong, or some images or labels are missing. Details in `./errors.json`")
    else:
        print("  [✓] No errors in the selected datasets")

        if args.draw:
            print("\nDraw labels")
            for dataset in datasets.keys():
                dataset_instance = datasets[dataset]
                draw_labels(dataset_instance.path())
        if args.generate:
            ocr_generator: Generator = OCR_SYSTEMS[ocr_system](
                test_name,
                list(datasets.keys()),
                augmentation_config
            )
            ocr_generator.generate_data(tasks)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Training data generator for Text Detection and Text Recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--draw", action="store_true", help="Draw labels")
    group.add_argument("--generate", action="store_true", help="Start training data generation")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    pipeline_config: dict = yaml.safe_load(open("./config/pipeline.yaml", "r"))
    augmentation_config: dict = yaml.safe_load(open("./config/augmentation.yaml", "r"))

    TASKS: dict = pipeline_config["tasks"]
    TEST_NAME: str = pipeline_config["test-name"]
    OCR_SYSTEM: str = pipeline_config["ocr-system"]

    datasets: dict = pipeline_config["datasets"]
    defined_classes = list(map(lambda x: str(x).lower(), list(DATASETS.keys())))
    config_classes = list(datasets.keys())

    if len(defined_classes) > len(config_classes):
        print("  [✗] There are some datasets that are not defined in the `pipeline.yaml`")
    if len(defined_classes) < len(config_classes):
        print("  [✗] Some datasets in the `config.json` do not have a script in `./src/dataset/`")

    selected_datasets = {}
    for dataset in config_classes:
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
        
    if len(selected_datasets.keys()) == 0:
        print("  [✗] Select at least one dataset. No dataset selected")
        exit()

    if not(args.draw) and not(any(value == "y" for value in TASKS.values())):
        print("  [✗] Select at least one task. All the tasks are set to 'n'")
        exit()

    pipeline(
        TEST_NAME,
        OCR_SYSTEM,
        TASKS,
        selected_datasets,
        augmentation_config,
        args
    )