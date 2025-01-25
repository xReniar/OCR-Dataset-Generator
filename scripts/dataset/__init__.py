import os
import importlib

current_dir = os.path.dirname(os.path.abspath(__file__))

dataset_files = [
    f[:-3] for f in os.listdir(current_dir)
    if f.endswith(".py") and f not in ("__init__.py", "dataset.py", "dataset_old.py")
]

DATASETS = {}
CONFIG = {}

for dataset_file in dataset_files:
    module = importlib.import_module(f".{dataset_file}", package=__name__)
    class_name = dataset_file.upper()

    if hasattr(module, class_name):
        DATASETS[class_name] = getattr(module, class_name)

    if hasattr(module, "CONFIG"):
        CONFIG[class_name] = getattr(module, "CONFIG")

__all__ = ["DATASETS", "CONFIG"]