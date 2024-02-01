import argparse
import json
import os


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--path", type=str, required=True, help="Path to directory or json-file"
    )
    parser.add_argument("-r", "--recursive", type=bool, default=False)

    return parser.parse_args()


def cut_image_data(fpath: str):
    data: dict = json.load(open(fpath, "r", encoding="utf-8"))
    data["imageData"] = None
    json.dump(data, open(fpath, "w", encoding="utf-8"), ensure_ascii=False)


def run(params):
    if os.path.isfile(params.path):
        cut_image_data(params.path)
    else:
        for root, dirs, files in os.walk(params.path):
            for fname in files:
                if fname.lower().endswith(".json"):
                    cut_image_data(os.path.join(root, fname))

            if not params.recursive:
                break


if __name__ == "__main__":
    # Пример запуска:
    # python labelme_data_cutter.py -p=/images

    run(parse())
