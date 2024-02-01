import json
import os
import re

import click
import numpy as np
import pandas as pd

from statistics import median
from tqdm import tqdm
from PIL import Image, ImageOps
from typing import Tuple, List, Dict, Set, Union, Optional
from hashlib import sha256
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from collections import defaultdict, Counter
from ulid import ULID
from shutil import copy
import imagehash

from utils.references import IMAGE_EXTENSIONS, SKIP_DIRS
from utils.logger import init_logger
from utils.readers import scan_dir

logger = init_logger("prepare_data")


def phash(image: Image.Image, hash_size: int) -> str:
    return str(sha256(bytearray(imagehash.phash(image, hash_size=hash_size).hash.flatten().astype(int).tolist())).hexdigest())


def dhash(image: Image.Image, hash_size: int) -> str:
    """
    Хэширование dhash с конкатенацией длины dhash для уменьшения размера итогового файла
    :param image: Изображение в цветовой схеме
    :param hash_size: Размер resize
    :return: Строка - хэш
    """
    image = image.resize((hash_size + 1, hash_size))
    image = np.array(image, dtype=np.uint8)
    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = image[:, 1:] > image[:, :-1]
    # convert the difference image to a hash and return it
    dhash_val = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    return f"{sha256(str(dhash_val).encode('utf-8')).hexdigest()}{len(str(dhash_val))}"


def stemming(image: Image.Image, hash_size: int) -> str:
    image = image.resize((hash_size, hash_size))
    image = np.array(image, dtype=np.uint8)
    new_img = []
    for pix in image.flatten().tolist():
        new_img.append(pix & int("11111100", 2))
    return str(sha256(bytearray(new_img)).hexdigest())


def mean_std(img_np: np.array, max_pix_value: float = 255.0) -> (List[float], List[float]):
    """
    MEAN и STD по изображению в каждом канале RGB
    :param img_np: numpy array RGB
    :param max_pix_value: Размерность
    :return: mean RGB, std RGB
    """

    mean_values = list()
    std_values = list()

    for idx in range(img_np.shape[2]):
        data = img_np[:, :, idx].reshape(-1)
        mean_values.append(np.mean(data) / max_pix_value)
        std_values.append(np.std(data) / max_pix_value)

    return mean_values, std_values


def get_image_info(params: tuple) -> Dict[str, Union[str, int]]:
    """
    Расчет инфы по изображению
    :param params: Путь к изображению, метод хэширования, размер хэш-изображения, True - рассчитать 4 хэша для одного изображения (поворот на 90)
    :return: Словарь со свойствами
    """

    img_hashes = defaultdict(list)
    fpath, hash_methods, hash_size, use_rotate = params
    file_name = os.path.basename(fpath)
    file_extension = file_name.rsplit(".", 1)[1].lower()

    with open(fpath, "rb") as f:
        # Работаем с байтами
        fbytes = f.read()
        file_hash = str(sha256(fbytes).hexdigest())
        file_size = len(fbytes)

        # Работаем с изображением
        f.seek(0)

        img = Image.open(f).convert("RGB").copy()

        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        img_height, img_width = img.height, img.width
        img_mean, img_std = mean_std(np.array(img, dtype=np.uint8))
        img = img.convert("L")

        for hash_method in hash_methods:
            img_hashes[hash_method.__qualname__].append(hash_method(img, hash_size))

        if use_rotate:
            for angle in (90, 180, 270):
                rotated_img = img.rotate(angle, expand=True)
                for hash_method in hash_methods:
                    img_hashes[hash_method.__qualname__].append(hash_method(rotated_img, hash_size))

    return {
        "file_name": file_name,
        "file_extension": file_extension,
        "file_path": fpath,
        "img_hashes": dict(img_hashes),
        "mean": img_mean,
        "std": img_std,
        "aspect_ratio": img_width / img_height,
        "height": img_height,
        "width": img_width,
        "size": file_size,
        "sha256": file_hash
    }


def remove_duplicates(all_instances: List[str], unique_instances: List[str], use_rotate: bool):
    """
    Удаление дубликатов
    :param all_instances: Пути ко всем файлам датасета
    :param unique_instances: Пути к уникальным файлам датасета
    :param use_rotate: True - добавить проверку для повернутых дубликатов
    :return:
    """

    n = 0
    duplicates = set(all_instances) - set(unique_instances)
    if use_rotate:
        rotated_duplicates = {fpath for fpath, count in Counter(unique_instances).items() if count < 4}
        duplicates = duplicates | rotated_duplicates

    for fpath in duplicates:
        if os.path.isfile(fpath):
            os.remove(fpath)
            n += 1

    logger.info(f"Количество дубликатов: {n}")


def save_metadata(instances: List[Dict], fpath_index: str, dpath_root: str):
    """
    Сохранение рассчитанной инфы в CSV
    :param instances: Иформация по изображениям
    :param fpath_index: Путь нового CSV
    :param dpath_root: Путь до директории датасета - нужен для расчета вложенности файлов
    :return:
    """

    df_insts = pd.DataFrame(instances)
    df_insts["directory"] = df_insts["file_path"].apply(
        lambda x: x.rsplit("/", 1)[0].replace(dpath_root, "")[1:]
    )
    df_insts.drop(columns=["file_path"], inplace=True)

    logger.info(rf"Сохранение подробной информации в: {fpath_index}")
    df_insts.sort_values(by=["directory", "file_name"], inplace=True)
    df_insts.to_csv(fpath_index, sep=",", encoding="utf-8", index=False)


def check_small_image(params: tuple) -> Optional[str]:
    fpath = params[0]
    size = params[1]
    with open(fpath, "rb") as f:
        img = Image.open(f).convert("RGB").copy()

    if img.width < size or img.height < size or img.height * img.width < size**2:
        return fpath


def remove_json_without_image(image_dir: str, annot_dir: str, recursive: bool):
    fpath_imgs = []
    fpath_annots = []
    re_abs_path = re.compile(r"^/")

    # Поиск изображений
    for root, _, files in os.walk(image_dir):
        fpath_imgs.extend([re_abs_path.sub("", os.path.join(root.replace(image_dir, ""), fname)) for fname in files if fname.lower().endswith(IMAGE_EXTENSIONS)])
        if not recursive:
            break

    # Поиск аннотаций
    for root, _, files in os.walk(annot_dir):
        fpath_annots.extend([re_abs_path.sub("", os.path.join(root.replace(annot_dir, ""), fname)) for fname in files if fname.lower().endswith((".json", ".txt"))])
        if not recursive:
            break

    # Определение аннотаций без изображений
    stem_fpath_imgs = {os.path.splitext(i)[0] for i in fpath_imgs}
    diff_annots = [
        annot
        for annot in fpath_annots
        if os.path.splitext(annot)[0] not in stem_fpath_imgs
    ]

    # Доп. проверка - если в файле аннотации labelme есть путь до изображения
    stay_annots = []
    for fpath in diff_annots:
        if not fpath.lower().endswith(".json"):
            continue

        with open(os.path.join(annot_dir, os.path.normpath(fpath)), "rb") as f:
            data = json.load(f)

        if "imagePath" not in data or os.path.isfile(os.path.join(annot_dir, os.path.split(fpath)[0], os.path.normpath(data["imagePath"]))):
            stay_annots.append(fpath)

    # Удаление аннотаций без изображений
    for fpath in tqdm(set(diff_annots) - set(stay_annots), desc="remove annotations without image"):
        os.remove(os.path.join(annot_dir, os.path.normpath(fpath)))


def _compare_images(image_info: dict, unique_instances: dict, duplicates: dict):
    for hash_method, hashes in image_info["img_hashes"].items():
        for hash_value in hashes:
            exist_item = unique_instances[hash_method].get(hash_value)
            if is_replaced := (exist_item is None or image_info["size"] > exist_item["size"]):
                unique_instances[hash_method][hash_value] = image_info

            # Если нашли дубликат - сохраняем связку дублей
            if exist_item:
                if is_replaced:
                    duplicates[hash_method][exist_item["file_path"]] = image_info["file_path"]
                else:
                    duplicates[hash_method][image_info["file_path"]] = exist_item["file_path"]


def calculate_image_info(fpaths: List[str], params: List[tuple], workers: int) -> (dict, dict):
    unique_instances = defaultdict(dict)
    duplicates = defaultdict(dict)

    # Рассчитываем инфу по изображениям
    process_bar = tqdm(total=len(fpaths), desc="calculation image info")
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for item in executor.map(get_image_info, params):
                _compare_images(item, unique_instances, duplicates)
                process_bar.update()
    else:
        for params_item in params:
            item = get_image_info(params_item)
            _compare_images(item, unique_instances, duplicates)
            process_bar.update()
    process_bar.close()

    return dict(unique_instances), dict(duplicates)


def copy_duplicates(source_dir: Path, duplicates_path: Path, duplicates: dict):
    def _collect(_duplicates: dict, _children: List[str], _parent: str) -> (str, List[str]):
        if _elem := _duplicates.pop(_parent, None):
            _children.append(_parent)
            return _collect(_duplicates, _children, _elem)
        else:
            return _parent, _children

    original_paths = {}

    for _, dupls in duplicates.items():
        for dpl, orig in dupls.items():
            parent, children = _collect(dupls, [dpl], orig)

            if (base_path := original_paths.get(parent)) is None:
                base_path = duplicates_path / str(ULID())
                original_paths[parent] = base_path

            original_dir = base_path / "original"
            duplicates_dir = base_path / "duplicates"
            os.makedirs(base_path, exist_ok=True)
            os.makedirs(original_dir, exist_ok=True)
            os.makedirs(duplicates_dir, exist_ok=True)

            for dir_path, elem in zip(
                [original_dir] + [duplicates_dir] * len(children),
                [parent] + children,
            ):
                dest_fpath = dir_path / Path(elem).relative_to(source_dir)
                dest_fpath.parent.mkdir(parents=True, exist_ok=True)
                copy(elem, dest_fpath)


@click.command()
@click.option("--source-dir", "-s", type=str, required=True, help="Path to dataset directory")
@click.option("--protect-dir", "-p", type=str, help="Path to protect directory (test datasets, e.t.c.)")
@click.option("--annotation-dir", "-a", type=str, default=None, show_default=True, help="Path to annotation directory")
@click.option("--recursive", "-r", is_flag=True, default=False, show_default=True, help="Recursive dataset traversal")
@click.option("--workers", "-w", type=int, default=2, show_default=True, help="Number of processes")
@click.option("--hash-size", "-h", type=int, default=8, show_default=True, help="Hash size")
@click.option("--less-than", "-lt", type=int, help="Minimum image size")
@click.option("--method", "-m", type=str, help="Only hash method")
@click.option("--no-rotate", "-nor", is_flag=True, default=False, show_default=True, help="Not use rotate for calculate image hash")
@click.option("--save-duplicates", "-d", type=str, default=None, show_default=True, help="Path to save duplicates")
@click.option("--csv-info", "-i", type=str, help="CSV metadata")
def run(
    source_dir: str,
    protect_dir: str = None,
    annotation_dir: str = None,
    recursive: bool = False,
    workers: int = 2,
    hash_size: int = 8,
    less_than: Optional[int] = None,
    method: str = None,
    no_rotate: bool = False,
    save_duplicates: Optional[str] = None,
    csv_info: Optional[str] = None
):
    duplicates_path = None
    if save_duplicates:
        duplicates_path = Path(save_duplicates) / str(ULID())
        os.makedirs(duplicates_path.parent, exist_ok=True)
        os.makedirs(duplicates_path)

    use_rotate = not no_rotate
    unique_instances = dict()
    protect_fpaths = []
    funcs = [func for func in [dhash, stemming, phash] if not method or func.__name__ == method]
    logger.info(rf"Обработка датасета: {source_dir}")

    if less_than:
        logger.info(f"===============================")
        logger.info(f"== RUN remove images less than {less_than}")
        logger.info(f"===============================")

        small_images = []
        fpaths = list(scan_dir(source_dir, recursive=recursive))
        params = [(fpath, less_than) for fpath in fpaths]
        process_bar = tqdm(total=len(fpaths), desc="check small images")
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for item in executor.map(check_small_image, params):
                    if item:
                        small_images.append(item)
                    process_bar.update()
        else:
            for params_item in params:
                item = check_small_image(params_item)
                if item:
                    small_images.append(item)
                process_bar.update()
        process_bar.close()

        for fpath in tqdm(small_images, desc="remove small images"):
            os.remove(fpath)

    logger.info(f"===============================")
    logger.info(f"== RUN {[i.__qualname__ for i in funcs]} with {hash_size=}")
    logger.info(f"===============================")

    # Сбор всех файлов для обработки
    fpaths = list(scan_dir(source_dir, recursive=recursive))
    params = [(fpath, funcs, hash_size, use_rotate) for fpath in fpaths]
    n_files = len(fpaths)

    if n_files:
        logger.info(rf"Количество файлов: {n_files}")
        unique_instances, duplicates = calculate_image_info(fpaths, params, workers)

        # Если передан защищенный датасет, в котором по-максимому нужно оставить
        if protect_dir:
            protect_fpaths = list(scan_dir(protect_dir, recursive=recursive))
            protect_params = [(fpath, funcs, hash_size, use_rotate) for fpath in protect_fpaths]
            logger.info(rf"Количество защищенных файлов: {len(protect_fpaths)}")
            protect_instances, _ = calculate_image_info(protect_fpaths, protect_params, workers)
            for hash_method, items in protect_instances.items():
                unique_instances[hash_method].update(items)

        # Сохранение дубликатов
        if duplicates_path and duplicates:
            copy_duplicates(Path(source_dir), duplicates_path, duplicates)

        logger.info(f"===============================")
        logger.info(f"Удаление дубликатов")
        logger.info(f"===============================")
        for hash_method, items in unique_instances.items():
            logger.info(hash_method)
            remove_duplicates(
                all_instances=fpaths,
                unique_instances=[i["file_path"] for i in items.values()],
                use_rotate=use_rotate,
            )

    for _, instances in unique_instances.items():
        exist_instances = list({i["file_path"]: i for i in instances.values() if os.path.isfile(i["file_path"])}.values())

        heights = [i["height"] for i in exist_instances]
        widths = [i["width"] for i in exist_instances]
        aspect_ratio = [i["aspect_ratio"] for i in exist_instances]

        median_height = int(median(sorted(heights)))
        median_width = int(median(sorted(widths)))
        median_aspect_ratio = median(sorted(aspect_ratio))

        logger.info(f"===============================")
        logger.info(f"Количество уникальных сэмплов: {len(exist_instances) - len(protect_fpaths)}")
        logger.info(rf"mean: {np.array([i['mean'] for i in exist_instances]).sum(axis=0) / len(exist_instances)}")
        logger.info(rf"std: {np.array([i['std'] for i in exist_instances]).sum(axis=0) / len(exist_instances)}")
        logger.info(rf"widths: min = {min(widths)}, max = {max(widths)}, median = {median_width}, mean = {sum(widths)//len(widths)}")
        logger.info(rf"heights: min = {min(heights)}, max = {max(heights)}, median = {median_height}, mean = {sum(heights)//len(heights)}")
        logger.info(rf"aspects ratio (w/h): median = {median_aspect_ratio}, mean = {sum(aspect_ratio)/len(aspect_ratio)}")

        # Создать csv
        if len(csv_info or "") > 1:
            save_metadata(exist_instances, csv_info, source_dir)

        break
    else:
        logger.info("Изображения не найдены")

    if annotation_dir:
        remove_json_without_image(source_dir, annotation_dir, recursive)


if __name__ == "__main__":
    # Пример запуска:
    # python prepare_data.py -s /home/some_dataset
    #
    # Дополнительные параметры:
    #  -p <str: Путь к файлам, которые по-максимому нужно оставить (например: тестовый датасет)>
    #  -w <int: Количество процессов>
    #  -a <bool: Если передан - удаление аннотаций (labelme, yolo) сегментации без изображений>
    #  -r <bool: Если передан - подготовка всех вложенных директорий датасета>
    #  -ur <bool: Если передан - расчет хэшей происходит для 4-х градусов каждого изображения>
    #  -h <int: hash_size>
    #  -lt <int: remove images with height/width less than size>
    #  -m <str: hash method - dpash, phash, stemming>
    #  -i <str: Создание CSV с метаданными датасета>
    run()
