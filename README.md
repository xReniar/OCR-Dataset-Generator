# OCR dataset generator
This project is a tool for downloading and managing `OCR` datasets, combining online and local sources. It supports the creation of training data for `text detection` and `text recognition`.

<img src="./docs/Pipeline.png" width=100%>

This are the supported `OCR` tools:
- `docTR`: https://github.com/mindee/doctr
- `mmOCR`: https://github.com/open-mmlab/mmocr
- `PaddleOCR`: https://github.com/PaddlePaddle/PaddleOCR
- `YOLO`: Coming Soon
- `TrOCR`: Coming Soon

At the moment the datasets that can be used to generate the training data are:
- `FUNSD`: https://guillaumejaume.github.io/FUNSD/
- `SROIE`: https://paperswithcode.com/paper/icdar2019-competition-on-scanned-receipt-ocr
- `XFUND`: https://github.com/doc-analysis/XFUND (`de`,`es`,`fr`,`it`,`ja`,`pt`,`zh`)

The datasets that will be added are:
- `IAM`: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
- `GNHK`: https://www.goodnotes.com/gnhk

## Setup
```shell
# clone the repository
git clone https://github.com/xReniar/OCR-Dataset-Generator.git

# install the requirements:
cd OCR-Dataset-Generator
pip3 install -r requirements.txt
```

## Generate training data
To generate the training data check the `./config/pipeline.yaml` first. This yaml file contains:
- `test-name`: the generated training data will be stored in `./output/{test-name}`
- `ocr-system`: the ocr system that will be trained, the choices are `doctr`, `mmocr`, `paddleocr`
- `tasks`: specify if the training data is for `detection`, `recognition` or both. Possible values are `y` or `n`:
  ```yaml
  # both detection and recognition data will be generated
  tasks:
    det: y
    rec: y
  ```
- `datasets`: specify which datasets are going to be used for the generation of the training data. To select the dataset just set it to `y` otherwise set it to `n`, example below:
  ```yaml
  # this example selects SROIE and XFUND-ES dataset and combines them
  sroie: y
  xfund:
    xfund-de: n
    xfund-es: y
  ```

After selecting the datasets and the task it's possible to start generating the training data by running `main.py`. The arguments that need to be passed are mutually exclusive and they are:
- `--generate`: starts the pipeline and stores the training data inside `./output/{test-name}`
- `--draw`: for each dataset folder inside `./data`, a draw folder will be created, containing two subfolders: `train` and `test`. These subfolders will store all images with the bounding boxes drawn.

```shell
# examples
python3 main.py --generate
python3 main.py --draw
```

The generation of the training data or the drawing process won't start unless this errors are not solved:
- label does not have a corresponding image
- image does not have a corresponsing label
- the values of bounding box are wrong

# Data output
Below are shown the output folder after the generation of the training data, and instruction on how to use them. The examples below suppose that both tasks are executed:
- [doctr output](./docs/output_doctr.md)
- [mmocr output](./docs/output_mmocr.md)
- [paddleocr output](./docs/output_paddleocr.md)
# Docs
- [Understand how datasets works](./docs/Dataset.md)
- [Add new Dataset](./docs/AddDataset.md)
- [Add new Generator](./docs/AddGenerator.md)

## Future developments
- Add workflow for data augmentation
- Modify dataset to manage rotated text