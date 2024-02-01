import os
from pathlib import Path
from zipfile import ZipFile

import click
import pandas as pd
from tqdm import tqdm

from handlers.data_packer.common import find_indexes
from handlers.utils.logger import init_logger

logger = init_logger("data unpacker")


@click.group()
def cli_unpacking():
    pass


def chunk_unpacking(arch_path: Path):
    if arch_path.is_file():
        with ZipFile(arch_path, "r") as zipf:
            zipf.extractall(arch_path.parent)
        os.remove(arch_path)


def get_archive_paths(index_path: Path):
    df = pd.read_csv(index_path, sep=",", encoding="utf-8")
    for i in sorted(set(df["archive"])):
        yield index_path.parent / i


def run(dir_path: Path):
    process_bar = tqdm()
    for index in find_indexes(dir_path):
        archive_paths = list(get_archive_paths(index))

        process_bar.reset(total=len(archive_paths))
        process_bar.set_description(desc=str(index.parent))

        for arch_path in archive_paths:
            chunk_unpacking(arch_path)
            process_bar.update()
    process_bar.close()


@cli_unpacking.command()
@click.option("--source-dir", "-s", help="Path to captcha directory")
def unpacking(source_dir: str):
    source_dir = Path(source_dir)
    run(source_dir)


__all__ = ["cli_unpacking"]
