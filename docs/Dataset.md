# Data folder structure
Inside the `data` folder, are stored every downloaded dataset specified in the `config.json`.  Below is shown the structure of the `data` folder:
```bash
.
└── data
    ├── dataset-A -----------> # type-1 
    │   ├── test
    │   │   ├── images
    │   │   └── labels
    │   └── train
    │       ├── images
    │       └── labels
    ├── dataset-B -----------> # type-2
    │   ├── sub-dataset-1 ---> # variant-1 of dataset B
    │   │   ├── test
    │   │   │   ├── images
    │   │   │   └── labels
    │   │   └── train
    │   │       ├── images
    │   │       └── labels
    │   └── sub-dataset-2 ---> # variant-2 of dataset B
    │       ├── test
    │       │   ├── images
    │       │   └── labels
    │       └── train
    │           ├── images
    │           └── labels
    └── ....
```
To understand how to add new datasets, it is necessary to distinguish between two categories of datasets.
- `type-1`: this is the default type of a dataset.
- `type-2`: this is just a collection of `type-1` datasets
  - `sub-dataset`: in the example above the `dataset-B` has more variants, each variant is a `sub-dataset`
## Dataset format
So in the example above the structure of `type-1` and `sub-dataset` are the same and explained below:
```bash
.
├── test # suppose from 100 to 150 are for the testing
│   ├── images
│   │   ├── img100.jpg
│   │   ├── img101.jpg
│   │   └── ...
│   │   └── img150.jpg
│   └── labels
│       ├── img100.txt
│       ├── img101.txt
│       ├── ...
│       └── img150.txt
└── train # suppose from 1 to 99 are for the training
    ├── images
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    └── labels
        ├── img1.txt
        ├── img2.txt
        └── ...
```
- the `train` and `test` folder have the same structure, they contain 2 folders: `images` and `labels`
- inside the `labels` folder there is a `.txt` file for each image present in the `images` folder. The name of each `.txt` file should be the same as the corresponding image as shown in the example above. Make sure that the label's name only has the image's name without the extension.

Each line of the `.txt` file contains the `bounding box` associated with each `word`, separated by `\t`. When generating your data you could use this command:
```py
# example on how to create .txt files
# word is a string and bbox is a list
file.write(f"{word}\t{bbox}\n")
```
The `.txt` file should look something like this:
```txt
This [x1,y1,x2,y2]
is [x1,y1,x2,y2]
an [x1,y1,x2,y2]
example [x1,y1,x2,y2]
...
```
Where `x1`,`y1`,`x2`,`y2` are absolute points:
- `x1`, `y1`: top-left coordinates
- `x2`, `y2`: bottom-right coordinates
# Dataset class
For every dataset that is created there is always a `.py` file inside `./src/dataset/`. The dataset can be `local` or `online`:
- `local`: the dataset you want to use for generating the training data is already in your pc, you just need to convert it in the specified [format](#dataset-format) 
- `online`: the annotations and images or only the annotations are available online, the `_download()` automatically downloads images and labels and stores them in the `./data` folder in the specified [format](#dataset-format)

This is an example of a `.py` file for a dataset:
```py
from .dataset import Dataset

CONFIG = {...}

class dataset(Dataset):
    def __init__(
        self,
        config: dict
    ) -> None:
        super().__init__(config)

    def _download(self) -> None:
        # download code
```
The `CONFIG` specify the download links for `images` and `labels`, defines also the structure inside the `./data/` folder:
  ```py
  # local dataset case
  CONFIG = {}

  # type-1 dataset case
  # if it's only one it's a type-1 dataset
  CONFIG = {
      "dataset-name": "..."
  }

  # type-2 dataset case
  # if it's more than one it's a type-2 dataset
  CONFIG = {
      "sub-1": "...",
      "sub-2": "...",
  }
  ```
- `dataset` class: takes only the `CONFIG` as input

More information can be found [here](./AddDataset.md) 