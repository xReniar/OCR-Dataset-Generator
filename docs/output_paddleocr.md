# PaddleOCR
This is the structure of the generated training data
```txt
.
└── output
    └── {test-name}-paddleocr
        ├── Detection
        │   ├── test
        │   ├── train
        │   ├── test_label.txt
        │   └── train_label.txt
        └── Recognition
            ├── test
            ├── train
            ├── test_label.txt
            └── train_label.txt
```
Download the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) repository before performing the instruction below.
## Text Detection training
- create a `./train_data` folder inside `PaddleOCR` and copy the `Detection` folder inside it
- after creating or selecting a `config.yaml` inside `PaddleOCR/configs/det/*`, search for this 4 fields and modify them with the correct path:
```yaml
Train:
    dataset:
        data_dir: ./train_data/Detection/
        label_file_list: ./train_data/Detection/train_label.txt

Eval:
    dataset:
        data_dir: ./train_data/Detection/
        label_file_list: ./train_data/Detection/test_label.txt
```

## Text Recognition training
- create a `./train_data` folder inside `PaddleOCR` and copy the `Recognition` folder inside it
- after creating or selecting a `config.yaml` inside `PaddleOCR/configs/rec/*`, search for this 4 fields and modify them with the correct path:
```yaml
Train:
    dataset:
        data_dir: ./train_data/Recognition/
        label_file_list:
            - ./train_data/Recognition/train_label.txt

Eval:
    dataset:
        data_dir: ./train_data/Recognition/
        label_file_list:
            - ./train_data/Recognition/test_label.txt
```