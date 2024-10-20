import json


def main(args):
    print(args)

if __name__ == "__main__":
    config = json.load(open("config/config.json"))
    