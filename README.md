# OCR dataset generator
Training data generator for Text Detection and Text Recognition. The training data will be generated following the format specified by the various supported OCR systems. The supported OCR systems are:

<p align="center">
    <a href="https://github.com/mindee/doctr">
        <img src="./icons/doctr.png" width="70">
    </a>
    <a href="https://github.com/open-mmlab/mmocr">
        <img src="./icons/mmocr.png" width="70">
    </a>
    <a href="https://github.com/PaddlePaddle/PaddleOCR">
        <img src="./icons/paddleocr.jpeg" width="70">
    </a>
</p>

At the moment the datasets that can be used to generate the training data are:
- `FUNSD`: https://guillaumejaume.github.io/FUNSD/
- `IAM`: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
- `SROIE`: https://paperswithcode.com/paper/icdar2019-competition-on-scanned-receipt-ocr
- `XFUND`: https://github.com/doc-analysis/XFUND (`de`,`es`,`fr`,`it`,`ja`,`pt`,`zh`)
# Generate training data
To generate the training data just execute:
```bash
bash run.sh
```

# Datasets
Inside the `data` folder, there are folders for every dataset specified in the `config.json`
```bash
.
└── data
    ├── dataset-A -----------> # type-1 
    │   ├── images
    │   ├── test
    │   └── train
    ├── dataset-B -----------> # type-2
    │   ├── sub-dataset-1 ---> # variant-1 of dataset B
    │   │   ├── images
    │   │   ├── test
    │   │   └── train
    │   └── sub-dataset-2 ---> # variant-2 of dataset B
    │       ├── images
    │       ├── test
    │       └── train
    └── ....
``` 
To understand how to add new datasets, it is necessary to distinguish between two categories of datasets.
- `type-1`: this type of dataset does not have a variant for example `IAM` and `SROIE`
- `type-2`: this type of dataset have multiple version, for example `XFUND` have more languages 
  - `sub-dataset`: using the `XFUND` example that have more languages, there is a `sub-dataset` for every language
## Add new Dataset (with no variant)
To add a new dataset, create a folder with the name of the dataset, this folder will contain the subfolders: `images`, `train`, and `test`
- inside `images` put all the images of the dataset
- inside `train` and `test` put for each image a corresponding `.txt` file that contains the bounding box of every word, every `.txt` file has the same name of the corresponding image:
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
Save `word` and `bbox` separated by `\t`:
- `bbox` is a `list` containing 4 coordinates: `[x,y,X,Y]`, where `(x,y)` are the top-left coordinates and `(X,Y)` are the bottom-right coordinates.
```py
file.write(f"{word}\t{bbox}\n") # example on how to create .txt files
```
After this create, inside `./scripts/dataset/` a `{DatasetName}`.py, and copy this:
```py
from dataset import Dataset

class DatasetName(Dataset): # change name of the class with the dataset name
    def __init__(self):
        super().__init__(
            local_datasets = [],
            online_datasets = {}
        )

    def download(self, dataset:str | None = None):
        pass
```
## Add new Dataset (with variant)
Do this if you want to add a new dataset with more variants or if you want to add more variants for a dataset that already have variants. If the dataset you want to add have multiple variants like `XFUND` create a main folder with inside every `variants` containing `image`, `train` and `test` folder.
```py
# Case with existing dataset that already have variants
from dataset import Dataset

class ExistingDataset(Dataset): # change name of the class with the dataset name
    def __init__(self):
        super().__init__(
            local_datasets = [
                "new-variant" # just add here the name of the variant
            ],
            online_datasets = {
                "variant1": "..."
                "variant2": "..."
            }
        )

    def download(self, dataset:str | None = None): 
        pass
```
In this case if your dataset is online you can add the `variant` in the `online_datasets` with the download link and implement the `download` function, if it's just a local dataset put it in `local_datasets`
```py
# Case with new dataset
from dataset import Dataset

class NewDataset(Dataset): # change name of the class with the dataset name
    def __init__(self):
        super().__init__(
            local_datasets = [
                "variant1",
                "variant2"
            ],
            online_datasets = {}
        )

    def download(self, dataset:str | None = None):
        pass
```
