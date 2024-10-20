# OCR dataset generator
Training data generator for Text Detection and Text Recognition. The training data will be generated following the format specified by the various supported OCR systems. The supposted OCR systems are:
- [docTR](https://github.com/mindee/doctr)
- [mmOCR](https://github.com/open-mmlab/mmocr)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

At the moment the datasets that can be used to generate the training data are `IAM`, `SROIE`, `FUNSD`
# How to use
The `main.py` needs a json file where all the configuration for the training data are specified.
```json
{
    "name": "output_folder_name",
    "task": "training_data_task",
    "datasets": [
        "dataset_1",
        "..."
    ]
}
```
To start the generation process just run:
```bash
python3 main.py --config config/config.json
```
# Adding new datasets