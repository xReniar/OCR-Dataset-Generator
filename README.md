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
- `SROIE`: https://paperswithcode.com/paper/icdar2019-competition-on-scanned-receipt-ocr
- `XFUND`: https://github.com/doc-analysis/XFUND (`de`,`es`,`fr`,`it`,`ja`,`pt`,`zh`)

The datasets that will be added are:
- `IAM`: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

## Setup
Install the requirements:
```shell
pip3 install -r requirements.txt
```

# Generate training data
To generate the training data check the `./config/config.json` first. This json file specifies:
- `output`: the output of the training data, stored in `./output/`
- `ocr-system`: the ocr system that will be trained, the choices are `doctr`, `mmocr`, `paddleocr`
- `tasks`: specify if the training data is for `detection`, `recognition` or both.
  ```py
  "tasks": ["det"]        # only det
  "tasks": ["rec"]        # only rec
  "tasks": ["det", "rec"] # both
  ```
- `datasets`: specify which datasets are going to be used for the generation of the training data. To select the dataset just set it to `y` otherwise set it to `n`, example below:
  ```py
  "dataset1": "y",        # selected
  "dataset2": {
      "sub1": "n",        # not selected
      "sub2": "y"         # selected
  }
  ```

When everything is set up just run:
```shell
python3 main.py --generate
```
The generation of the training data won't start unless this errors are not solved:
- label does not have a corresponding image
- the values of bounding box are wrong

To check if the bounding boxes are correct run
```shell
python3 main.py --draw
```
This command creates a `draw` folder inside the dataset folder, where all the bounding boxes in the `train` and `test` folder are drawn.

# Data output
Below are shown the output folder after the generation of the training data, and instruction on how to use them. The examples below suppose that both tasks are executed:
- [doctr output](./docs/output_doctr.md)
- [mmocr output](./docs/output_mmocr.md)
- [paddleocr output](./docs/output_paddleocr.md)
# Docs
- [Understand how datasets works](./docs/Dataset.md)
- [Add new Dataset](./docs/AddDataset.md)
- [Add new Generator](./docs/AddGenerator.md)