import argparse

def main(args):
    print(args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for generating training data for Text Detection and Text Recognition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--config", type=str, required=True, help="path to config.json")
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)