from src.dataset import init_datasets
from src.generator import *
from src.utils.draw import draw_labels
from src.utils. dataset import check_images, check_labels
from colorama import Fore
import argparse
import os
import json
import yaml


def pipeline(
    test_name: str,
    ocr_system: str,
    tasks: dict,
    datasets: dict,
    lang: list[str] | None,
    workers: int,
    augmentation: bool,
    draw_args: dict,
    args
) -> None:
    print(f"{Fore.LIGHTCYAN_EX}[Dataset Setup]{Fore.RESET}")
    init_datasets(datasets)

    # create dataset objects
    for dataset in datasets.keys():
        # dataset setup
        datasets[dataset].setup()

    # check if all the labels have a corresponding image and viceversa
    # also check if there are wrong bounding boxes
    if os.path.isfile("./errors.json"):
        os.remove("./errors.json")

    print(f"\n{Fore.LIGHTCYAN_EX}[Error Checking step]{Fore.RESET}")
    errors = {}
    for dataset in datasets.keys():
        dataset_instance = datasets[dataset]
        missing = check_images(dataset_instance.path())
        missing_images = missing["missing_images"]
        missing_labels = missing["missing_labels"]

        label_errors = check_labels(dataset_instance.path())

        if not(len(missing_images) == 0 and len(missing_labels) == 0 and len(label_errors.keys()) == 0):
            errors[dataset_instance.__str__()] = dict(
                missing_images = missing_images,
                missing_labels = missing_labels,
                label_checking = label_errors
            )

    if len(list(errors.keys())) != 0:
        with open("errors.json", "w") as error_file:
            json.dump(errors, error_file, indent=4, ensure_ascii=False)
        print(f"{Fore.RED}✗{Fore.RESET} Some bbox values are wrong, or some images or labels are missing. Details in {Fore.RED}./errors.json{Fore.RESET}")
    else:
        print(f"{Fore.GREEN}✓{Fore.RESET} No errors in the selected datasets\n")

        if args.draw:
            draw_labels(
                datasets = sorted(list(datasets.keys())),
                lang = lang,
                workers = workers,
                color = draw_args["color"],
                thickness = draw_args["thickness"]
            )
        if args.generate:
            ocr_generator: Generator = OCR_SYSTEMS[ocr_system](
                test_name = test_name,
                datasets = list(datasets.keys()),
                dict = lang,
                workers = workers,
                augmentation = augmentation
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
    pipeline_config: dict = yaml.safe_load(open("pipeline.yaml", "r"))

    TASKS: dict = pipeline_config["tasks"]
    TEST_NAME: str = pipeline_config["test-name"]
    OCR_SYSTEM: str = pipeline_config["ocr-system"]
    dict_path: str | None = pipeline_config["dict"]
    WORKERS: int = pipeline_config["workers"]   
    DICT = list(map(lambda x: x.split("\n")[0], open(dict_path, "r").readlines())) if dict_path != None else None
    AUGMENTATION: bool = pipeline_config["augmentation"]

    datasets: dict = pipeline_config["datasets"]
    config_classes = list(datasets.keys())

    selected_datasets = {}
    for dataset in config_classes:
        value = datasets[dataset]
        if isinstance(value, str):
            if value == "y":
                selected_datasets[dataset] = None
        if isinstance(value, dict):
            for sub in value.keys():
                if value[sub] == "y":
                    selected_datasets[sub] = None
        
    args = parse_args()
        
    if len(selected_datasets.keys()) == 0:
        print(f"{Fore.RED}✗{Fore.RESET} Select at least one dataset. No dataset selected")
        exit()

    if not(args.draw) and not(any(value == "y" for value in TASKS.values())):
        print(f"{Fore.RED}✗{Fore.RESET} Select at least one task. All the tasks are set to None")
        exit()

    if WORKERS < 1:
        print(f"{Fore.RED}✗{Fore.RESET} The number of workers must be greater than 0")
        exit()

    pipeline(
        test_name = TEST_NAME,
        ocr_system = OCR_SYSTEM,
        tasks = TASKS,
        datasets = selected_datasets,
        lang = DICT,
        workers = WORKERS,
        augmentation = AUGMENTATION,
        draw_args = pipeline_config["draw-process"],
        args = args
    )