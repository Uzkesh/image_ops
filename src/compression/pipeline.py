import os

import click
import numpy as np
import albumentations as A
import cv2
from typing import List
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from shutil import copy
from PIL import Image

from utils.scanners import scan_img_json


def _brute_force_compress(
    np_img: np.ndarray,
    dest_fpath: Path,
    max_filesize: int,
    qualities: List[int],
):
    for new_quality in qualities:
        cv2.imwrite(
            str(dest_fpath.absolute()),
            np_img,
            [int(cv2.IMWRITE_JPEG_QUALITY), new_quality],
        )
        if dest_fpath.stat().st_size < max_filesize:
            break


def _resize_and_compress(fpath: Path, dest_fpath: Path, max_height: int, max_width: int, max_filesize: int) -> bool:
    qualities = [100, 95, 90, 85, 80, 75, 70, 65, 60]
    dest_fpath.parent.mkdir(parents=True, exist_ok=True)
    is_resized = False
    is_compressed = False

    if max_height and max_width:
        with open(fpath, "rb") as f:
            img = Image.open(f).convert("RGB").copy()

        if is_resized := img.height * img.width > max_height * max_width:
            img.thumbnail(
                size=(max_width, max_height) if img.width >= img.height else (max_height, max_width),
                resample=Image.LANCZOS,
            )
            np_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(dest_fpath.absolute()), np_img)

    if max_filesize:
        if is_resized:
            if is_compressed := dest_fpath.stat().st_size > max_filesize:
                if dest_fpath.suffix.lower() not in (".jpg", ".jpeg",):
                    # Если уменьшенное изображение не jpg и больше max_filesize, то
                    #  будем сжимать его и сохранять в jpg
                    os.remove(dest_fpath)
                    dest_fpath = dest_fpath.parent / f"{dest_fpath.stem}.jpg"

                _brute_force_compress(np_img, dest_fpath, max_filesize, qualities)
        else:
            if is_compressed := fpath.stat().st_size > max_filesize:
                np_img = A.read_bgr_image(str(fpath.absolute()))
                _brute_force_compress(np_img, dest_fpath, max_filesize, qualities)

    return is_resized or is_compressed


def image_compress(source: Path, destination: Path, workers: int, max_height: int, max_width: int, max_filesize: int):
    fpaths, jpaths, dest_fpaths, dest_jpaths = [], [], [], []

    for fpath, jpath in scan_img_json(source, recursive=True):
        fpaths.append(fpath)
        jpaths.append(jpath)
        dest_fpaths.append(destination / fpath.relative_to(source))
        dest_jpaths.append(destination / jpath.relative_to(source))

    process_bar = tqdm(total=len(fpaths))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for is_changed in executor.map(
                _resize_and_compress,
                fpaths,
                dest_fpaths,
                [max_height] * len(fpaths),
                [max_width] * len(fpaths),
                [max_filesize] * len(fpaths),
            ):
                if is_changed:
                    process_bar.update()
    else:
        for fpath, dest_fpath in zip(fpaths, dest_fpaths):
            if _resize_and_compress(fpath, dest_fpath, max_height, max_width, max_filesize):
                process_bar.update()
    process_bar.close()

    for jpath, dest_jpath in zip(jpaths, dest_jpaths):
        if jpath.is_file() and jpath.absolute() != dest_jpath.absolute():
            copy(jpath, dest_jpath)


@click.command()
@click.option("--source", "-s", type=str, help="Source directory")
@click.option("--destination", "-d", type=str, help="Destination directory")
@click.option("--workers", "-w", type=int, default=1, help="Number workers")
@click.option("--max-height", "-mh", type=int, default=3024, help="Max image height")
@click.option("--max-width", "-mw", type=int, default=4032, help="Max image width")
@click.option("--max-filesize", "-mf", type=int, default=2*1024**2, help="Max filesize")
def _run(mode: str, source: str, destination: str, workers: int, max_height: int, max_width: int, max_filesize: int):
    if mode == "large_image_resize":
        large_image_resize(Path(source), Path(destination), 8, (3024, 4032))
    elif mode == "large_image_compress":
        large_image_compress(Path(source), Path(destination), 8, 2 * 1024 ** 2)


if __name__ == "__main__":
    _run()
