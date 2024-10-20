import argparse

def main(args):
    print(args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for generating training data for Text Detection and Text Recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--task", type=str, required=True, help="task used to generate the traning data")
    parser.add_argument("--ocr", type=str, required=True, help="OCR system that will be used for training")
    parser.add_argument("--datasets", type=str, nargs="+", required=True, help="datasets that are needed to generate the traning data")
    parser.add_argument("--name", type=str, required=True, help="name of the folder where the traning data will be stored")
    args = parser.parse_args()

    # convert each dataset in the list to lowercase
    args.datasets = [dataset.lower() for dataset in args.datasets]
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)