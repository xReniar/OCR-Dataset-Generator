# Data folder structure
Inside the `data` folder, are stored every downloaded dataset specified in the `config.json`.  Below is shown the structure of the `data` folder:
```bash
.
└── data
    ├── dataset-A -----------> # type-1 
    │   ├── images
    │   ├── test
    │   └── train
    ├── dataset-B -----------> # type-2
    │   ├── sub-dataset-1 ---> # variant-1 of dataset B
    │   │   ├── images
    │   │   ├── test
    │   │   └── train
    │   └── sub-dataset-2 ---> # variant-2 of dataset B
    │       ├── images
    │       ├── test
    │       └── train
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
├── images
│   ├── img1.jpg # suppose img1 is for train
│   ├── img2.jpg # suppose img2 is for test
│   └── ...
├── test
│   ├── img2.txt # label for img2
│   └── ...
└── train
    ├── img1.txt # label for img1
    └── ...
```
- in the `images` folder are stored all the images of the dataset.
- inside `train` and `test` folder are stored `.txt` files where each `.txt` file have a corresponding image. The name of the `.txt` file should be the same as the corresponding image as shown in the example above.

These `.txt` files contains for every `word` the `bounding-box` associated, in the python script just do this:
```py
file.write(f"{word}\t{bbox}\n") # example on how to create .txt files
```
The `.txt` file should look something like this:
```txt
This [x1,y1,x2,y2]
is [x1,y1,x2,y2]
an [x1,y1,x2,y2]
example [x1,y1,x2,y2]
```
Where `x1`,`y1`,`x2`,`y2` are absolute points:
- `x1`,`y1`: top-left coordinates
- `x2`,`y2`: bottom-right coordinates
# Dataset class
For every dataset that is created there is always a `.py` file inside `./src/dataset/`. The dataset can be `local` or `online`:
- `local`: the dataset you want to use for generating the training data is already in your pc, you just need to convert it in the specified [format](#dataset-format) 
- `online`: the annotations and images or only the annotations are available online, the `download()` automatically downloads images and labels and stores them in the `./data` folder in the specified [format](#dataset-format)

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

    def download(self):
        super().download()
        # code
```
- `CONFIG`: specify the download links for `images` and `labels`, defines also the structure inside the `./data/` folder:
  ```py
  # local dataset case
  CONFIG = {}

  # type-1 dataset case
  CONFIG = {
      "dataset-name": "..."
  }

  # type-2 dataset case
  CONFIG = {
      "sub-1": "...",
      "sub-2": "...",
  }
  ```
- `dataset` class: takes only the `CONFIG` as input

More information can be found [here](./AddDataset.md) 