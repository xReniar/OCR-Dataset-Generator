# Add new Dataset
There are 2 possible types of dataset that is possible to add: `local` and `online`. In this tutorial suppose the name of the dataset you want to add is `custom`.

## Add local dataset
If you want to add `local` dataset follow this steps:

Create manually a `custom` folder inside `./data/`, it should look like this:
```bash
.
└── data
    ├── custom -----------> # type-1 case
    │   ├── test
    │   │   ├── images
    │   │   └── labels
    │   └── train
    │       ├── images
    │       └── labels
    ├── custom -----------> # type-2 case
    │   ├── sub-dataset-1 ---> # variant-1 of custom
    │   │   ├── test
    │   │   │   ├── images
    │   │   │   └── labels
    │   │   └── train
    │   │       ├── images
    │   │       └── labels
    │   └── sub-dataset-2 ---> # variant-2 of custom
    │   │   ├── test
    │   │   │   ├── images
    │   │   │   └── labels
    │   │   └── train
    │   │       ├── images
    │   │       └── labels
    └── ....
```
- Inside the `train` and `test` folder put the required files explained [here](Dataset.md#dataset-format).

- After this, the next steps are the same explained [here](#add-online-dataset) except:
  - the `CONFIG` is empty
  - the `_download()` function should be empty because it's a local dataset

## Add online dataset
If you want to add `online` dataset follow this steps:

Add a `custom.py` inside `./src/dataset/`. (It is mandatory that the character `-` should not be present in the dataset name). After creating the `.py` paste this:
```py
from .dataset import Dataset

CONFIG = { }   # [1]

class CUSTOM(Dataset):     # [2]
    def __init__(
        self,
        config: dict
    ) -> None:
        super().__init__(config)

    def _download(self):     # [3]
        # download code
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
- `[3]`: the dataset is online so the labels and images are stored online. Implement the `_download()` function that stores the data in the necessary [format](Dataset.md#data-folder-structure) inside the `./data/` folder

## Update pipeline.yaml
After creating the dataset class update the `./config/pipeline.yaml` to add the `custom` dataset, notice that it follows the same structure of the `CONFIG` file, so you can just copy it and modify the links with `y` or `n`.
```yaml
datasets:
    funsd: y
    iam:
    custom: y # set it to 'y' to include it in the generation process
```