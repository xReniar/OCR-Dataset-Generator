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
- Clone the [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) repository before performing the instruction below.
- Create a `./train_data` folder inside `.PaddleOCR` and put inside the `{test-name}-paddleocr` folder.
## Text Detection training
- after creating or selecting a `config.yaml` inside `PaddleOCR/configs/det/*`, search for this 4 fields and modify them with the correct path:
```yaml
Train:
    dataset:
        data_dir: ./train_data/{test-name}-paddleocr/
        label_file_list: ./train_data/{test-name}-paddleocr/Detection/train_label.txt

Eval:
    dataset:
        data_dir: ./train_data/{test-name}-paddleocr/
        label_file_list: ./train_data/{test-name}-paddleocr/Detection/test_label.txt
```
And to start the training:
```bash
cd PaddleOCR
# put the correct configuration file 
python3 tools/train.py -c configs/det/"selected_config".yml
```
- if the selected `.yaml` config is among the available ones, start the training directly with: 
```bash
# put the correct configuration file
# replace "{test-name}" with the correct folder name 
cd PaddleOCR
python3 tools/train.py \
    -c configs/det/"selected_config".yml \
    -o Train.dataset.data_dir=./train_data/{test-name}-paddleocr \
       Train.dataset.label_file_list=./train_data/{test-name}-paddleocr/Detection/train.txt \
       Eval.dataset.data_dir=./train_data/{test-name}-paddleocr \
       Eval.dataset.label_file_list=./train_data/{test-name}-paddleocr/Detection/test.txt
```

## Text Recognition training
- after creating or selecting a `config.yaml` inside `PaddleOCR/configs/rec/*`, search for this 4 fields and modify them with the correct path:
```yaml
Train:
    dataset:
        data_dir: ./train_data/{test-name}-paddleocr/
        label_file_list: ./train_data/{test-name}-paddleocr/Recognition/train_label.txt

Eval:
    dataset:
        data_dir: ./train_data/{test-name}-paddleocr/
        label_file_list: ./train_data/{test-name}-paddleocr/Recognition/test_label.txt
```
And to start the training:
```bash
cd PaddleOCR
# put the correct configuration file
python3 tools/train.py -c configs/rec/"selected_config".yml
```
- if the selected `.yaml` config is among the available ones, start the training directly with:
```bash
# put the correct configuration file 
# replace "{test-name}" with the correct folder name
cd PaddleOCR
python3 tools/train.py \
    -c configs/rec/"selected_config".yml \
    -o Train.dataset.data_dir=./train_data/{test-name}-paddleocr \
       Train.dataset.label_file_list=./train_data/{test-name}-paddleocr/Recognition/train.txt \
       Eval.dataset.data_dir=./train_data/{test-name}-paddleocr \
       Eval.dataset.label_file_list=./train_data/{test-name}-paddleocr/Recognition/test.txt
```