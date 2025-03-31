# YOLO + TrOCR
This is the structure of the generated training data
```txt
.
└── output
    └── {test-name}-yolotrocr
        ├── Detection
        │   ├── test
        │   │   ├── images
        │   │   └── labels
        │   └── train
        │       ├── images
        │       └── labels
        └── Recognition
            ├── test
            ├── train
            ├── test.txt
            └── train.txt
```
## Text Detection training
For more information click [here](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

To train yolo clone the `yolov5` repository:
```bash
git clone https://github.com/ultralytics/yolov5.git
```

Suppose the dataset is called `dataset`, create a `dataset.yaml` inside `yolov5/data` like this:
```yaml
path: path/to/output/{test-name}-yolotrocr/Detection
train: train/images
val: test/images

names:
   0: text
```

To start the training execute this commands:
```bash
cd yolov5
# suppose the selected model is the yolov5s
python3 train.py --epochs 3 --data dataset.yaml --weights yolov5s.pt
```
## Text Recognition training
Detailed instruction can be found here: [Fine Tuning TrOCR – Training TrOCR to Recognize Curved Text](https://learnopencv.com/fine-tuning-trocr-training-trocr-to-recognize-curved-text/)