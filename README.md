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

# Tutorial
- [Dataset](./docs/Dataset.md)
- [Add new Dataset](./docs/AddDataset.md)