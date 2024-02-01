import os
import click
from typing import Tuple, Set, Optional, Union, Generator
from pathlib import Path

from utils.references import IMAGE_EXTENSIONS, SKIP_DIRS


def scan_dir(
    dir_path: Union[Path, str],
    extensions: Optional[Tuple] = IMAGE_EXTENSIONS,
    skip_dirs: Optional[Set] = SKIP_DIRS,
    recursive: bool = False,
) -> Generator[Path, None, None]:
    """
    Получение списка файлов внутри директории

    :param dir_path: Директория
    :param extensions: интересующие расширения; все файлы в случае None
    :param skip_dirs: игнорируемые директории; все директории в случае None
    :param recursive: True - рекурсивных обход всех вложенных директорий
    :return: Полный путь к файлу
    """

    for root, _, files in os.walk(dir_path):
        proot = Path(root)

        if set(proot.parts) & skip_dirs:
            continue

        for fname in files:
            fpath = proot / fname
            if extensions is None or fpath.suffix.lower() in extensions:
                yield fpath

        if not recursive:
            break


def scan_img_json(
    dir_path: Union[Path, str],
    extensions: Optional[Tuple] = IMAGE_EXTENSIONS,
    skip_dirs: Optional[Set] = SKIP_DIRS,
    recursive: bool = False,
) -> Generator[Tuple[Path, Optional[Path]], None, None]:
    for fpath in scan_dir(
        dir_path=dir_path,
        extensions=extensions,
        skip_dirs=skip_dirs,
        recursive=recursive,
    ):
        jpath = fpath.parent / f"{fpath.stem}.json"
        yield fpath, jpath


@click.command()
@click.option()
def _run():
    pass


if __name__ == "__main__":
    _run()
