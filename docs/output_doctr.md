# docTR
This is the structure of the generated training data (suppose that both `detection` and `recognition` are selected)
```txt
.
└── output
    └── {test-name}-doctr
        ├── Detection
        │   ├── test
        │   │   ├── images
        │   │   └── labels.json
        │   └── train
        │       ├── images
        │       └── labels.json
        └── Recognition
            ├── test
            │   ├── images
            │   └── labels.json
            └── train
                ├── images
                └── labels.json
```
Download the [docTR](https://github.com/mindee/doctr) repository before performing the instruction below.
## Text Detection training
To start the training just run `train_pytorch.py` or `train_tensorflow.py` situated in `doctr/references/detection`, 
```bash
# this example shows training db_resnet50 with pytorch
python3 doctr/references/detection/train_pytorch.py \
    db_resnet50 \
    --train_path path/to/output/{test-name}-doctr/Detection/train \
    --val_path path/to/output/{test-name}-doctr/Detection/test \
    --epochs 5
```
## Text Recognition training
To start the training just run `train_pytorch.py` or `train_tensorflow.py` situated in `doctr/references/recognition`, 
```bash
# this example shows training crnn_vgg16_bn with pytorch
python3 doctr/references/recognition/train_pytorch.py \
    crnn_vgg16_bn \
    --train_path path/to/output/{test-name}-doctr/Recognition/train \
    --val_path path/to/output/{test-name}-doctr/Recognition/test \
    --epochs 5
```