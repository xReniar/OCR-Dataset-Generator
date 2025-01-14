import json
import argparse


def main(args):
    print(args)

def parse_args():
    config:dict = json.load(open("config/config.json", "r"))
    DATASETS = config["datasets"].keys()
    OCR_SYSTEMS = config["ocr-system"]

    parser = argparse.ArgumentParser(
        description="Training data script generator for text detection and text recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--name", type=str, required=True, default="custom_data", help="name of the training data")
    parser.add_argument("--task", type=str, required=True, choices=["det", "rec"], help="task for the training data")
    parser.add_argument("--ocr-system", type=str, required=True, choices=OCR_SYSTEMS)
    parser.add_argument("--datasets", type=str, required=True, default=DATASETS, help="datasets that are going to be used to generate the training data")
    parser.add_argument("--blur", dest="blur", action="store_true", help="add blurring to training data")
    parser.add_argument("--skew", dest="skew", action="store_true", help="add skewing to training data")
    parser.add_argument("--distort", dest="distort", action="store_true", help="add distortion to training data")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
    