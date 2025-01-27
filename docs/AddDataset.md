# Add new Dataset
There are 2 possible types of dataset that is possible to add: `local` and `online`. In this tutorial suppose the name of the dataset you want to add is `custom`.

## Add local dataset
If you want to add `local` dataset follow this steps:

Create manually a `custom` folder inside `./data/`, it should look like this:
```bash
.
└── data
    ├── custom -----------> # type-1 case
    │   ├── images
    │   ├── test
    │   └── train
    ├── custom -----------> # type-2 case
    │   ├── sub-dataset-1 ---> # variant-1 of custom
    │   │   ├── images
    │   │   ├── test
    │   │   └── train
    │   └── sub-dataset-2 ---> # variant-2 of custom
    │       ├── images
    │       ├── test
    │       └── train
    └── ....
```
- Inside the `images`, `train` and `test` folder put the required files explained [here](Dataset.md#dataset-format).

- After this, the next steps are the same explained [here](#add-online-dataset) except:
  - the `CONFIG` only specify the structure of the dataset and not the download links.
  - the `download()` function can be empty since it's a local dataset

## Add online dataset
If you want to add `online` dataset follow this steps:

Add a `custom.py` inside `./scripts/dataset/`. (It is mandatory that the character `-` should not be present in the dataset name). After creating the `.py` paste this:
```py
from .dataset import Dataset

CONFIG = { }   # [1]

class CUSTOM(Dataset):     # [2]
    def __init__(
        self,
        config: dict
    ) -> None:
        super().__init__(config)

    def download(self):     # [3]
        super().download()
```
- `[1]`: the `CONFIG` specifies the type of the dataset and the structure of it. Notice that if the dataset is a `type-2` the character `-` is used to distinguish between the various sub-datasets. The value of the keys can be anything, just follow the 2 examples below:
```py
# if `custom` is a `type-1` dataset
CONFIG = {
    "custom": "download link"
}

# if `custom` is a `type-2` dataset
# name convention -> "{dataset name}-{sub-dataset name}"
CONFIG = {
    "custom-sub1": "...."
    "custom-sub2": "...."
}
```
- `[2]`: the class name must be in uppercase
- `[3]`: the dataset is online so the labels and images are stored online. Implement the `download()` function that stores the data in the necessary [format](Dataset.md#data-folder-structure) inside the `./data/` folder

## Update config.json
After creating the dataset class update the `./config/config.json` to add the `custom` dataset, notice that if follow the same structure of the `CONFIG` file, so you can just copy it and modify the links with `y` or `n`.
```json
{
    "datasets": {
        "custom": "y",          # type-1 case
        "custom": {             # type-2 case
            "custom-sub1": "y",
            "custom-sub2": "n"
        }
    }
}
```