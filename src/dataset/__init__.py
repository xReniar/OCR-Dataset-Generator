from .dataset import LocalDataset, Dataset
import importlib
import os

current_dir = os.path.dirname(os.path.abspath(__file__))


def init_datasets(selected: dict[str, object]):
    selected_datasets = set(selected.keys())
    dataset_files = [
        f[:-3] for f in os.listdir(current_dir)
        if f.endswith(".py") and f not in ("__init__.py", "dataset.py", "_local.py", "_online.py")
    ]

    for dataset in selected_datasets:
        package_name:str = dataset.split("-")[0] if "-" in dataset else dataset

        if package_name in dataset_files:
            module = importlib.import_module(f".{package_name}", package=__name__)
            class_name = package_name.upper()

            CLASS = getattr(module, class_name)
            CONF: dict = getattr(module, "CONFIG")

            if "-" in dataset:
                dataset_obj: Dataset = CLASS(CONF)
                dataset_obj.set_subdataset(dataset)
                selected[dataset] = dataset_obj
            else:
                dataset_obj: Dataset = CLASS(CONF)
                selected[dataset] = dataset_obj
        else:
            selected[dataset] = LocalDataset(dict(__name__ = dataset))

__all__ = [
    init_datasets,
]