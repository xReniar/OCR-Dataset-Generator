from scripts.dataset import *
import json

__OCR_SYSTEMS__ = ["doctr", "mmocr", "paddleocr"]

def download_data(
    requested_datasets: list[Dataset],
    sub_datasets: dict
) -> None:
    for dataset in requested_datasets:
        if dataset.has_variants():
            for sub in sub_datasets[dataset._root_name()]:
                dataset.set_to(sub)
                if not(dataset.is_downloaded()):
                    dataset.download()
        else:
            if not(dataset.is_downloaded()):
                    dataset.download()

def main(
    test_name: str, 
    ocr_system: str,
    task: list,
    datasets
) -> None:
    # get root datasets (without the '-' after every name)
    main_datasets = set()
    sub_datasets = dict()
    for dataset in datasets:
        if isinstance(dataset, str):
            main_datasets.add(dataset)
        if isinstance(dataset, list):
            root, _ = dataset[0].split("-")
            sub_datasets[root] = dataset
            main_datasets.add(root)
    main_datasets: list[str] = list(main_datasets)

    # generating instances of datasets (Dataset object)
    requested_datasets:list[Dataset] = []
    for dataset in main_datasets:
        name_upper = dataset.upper()
        requested_datasets.append(DATASETS[name_upper](CONFIG[name_upper]))

    # download data if necessary
    download_data(requested_datasets, sub_datasets)


if __name__ == "__main__":
    config:dict = json.load(open("config/config.json", "r"))
    datasets:dict = config["datasets"]

    defined_classes = list(map(lambda x: str(x).lower(), list(DATASETS.keys())))
    config_classes = list(datasets.keys())
    if len(defined_classes) > len(config_classes):
        raise BaseException("There are some datasets that are not in the `config.json`")
    if len(defined_classes) < len(config_classes):
        raise BaseException("Some datasets in the `config.json` do not have a script in `./scripts/dataset/`")

    selected = []
    for dataset in list(datasets.keys()):
        value = datasets[dataset]
        if isinstance(value, str):
            selected.append(dataset) if value == "y" else None
        elif isinstance(value, dict):
            sub_list = []
            for sub in value.keys():
                sub_list.append(sub) if value[sub] == "y" else None
            selected.append(sub_list) if len(sub_list) > 0 else None
        else:
            raise ValueError(f"Type '{type(value)}' not available")
        
    main(
        config["test-name"],
        config["ocr-system"],
        config["task"],
        selected
    )
    