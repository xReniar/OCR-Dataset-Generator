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

| Dataset | Annotations | Images |
|---------|-------------|--------|
| `FUNSD` | Available   | Available |
| `IAM`   | Available   | Not available |
| `SROIE` | Available   | Not available |
| `XFUND` | Available   | Available |

# Generate training data
To generate the training data just execute:
```bash
bash run.sh
```

# Dataset
Inside the `data` folder, there are folders for every dataset specified in the `config.json`
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
- `type-1`: this is the default type of a dataset
- `type-2`: this is just a collection of `type-1` datasets
  - `sub-dataset`: in the example above the `dataset-B` has more variants, each variant is a `sub-dataset`
## Dataset format
So in the example above the structure of `type-1` and `sub-dataset` are the same and explained in the example below:
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
- insde `images` folder are stored all the images of the dataset
- inside `train` and `test` folder are stored `.txt` files where each `.txt` file have a corresponding image.

These `.txt` files contains for every `word` the `bounding-box` associated:
```py
file.write(f"{word}\t{bbox}\n") # example on how to create .txt files
```