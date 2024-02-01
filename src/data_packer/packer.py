import os
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Optional
from zipfile import ZipFile
from collections import defaultdict

import click
import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm
from ulid import ULID

from handlers.data_packer.common import INDEX_FILENAME, find_indexes
from handlers.utils.logger import init_logger
from handlers.utils.readers import scan_dir

logger = init_logger("data packer")


@click.group()
def cli_packing():
    pass


class FileDTO(BaseModel):
    filename: str
    size: int
    hash: str
    archive: Optional[str] = None


class RelativeDirDTO(BaseModel):
    dir: Path
    files: Dict[str, FileDTO] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class ChunksDTO(BaseModel):
    df: Optional[pd.DataFrame] = None
    dir: Path
    archives: Dict[str, List[FileDTO]] = Field(default_factory=lambda: defaultdict(list))

    class Config:
        arbitrary_types_allowed = True


class DataPacker:
    def __init__(self, source_dir: Path):
        self.source_dir = source_dir
        self.dataset: Dict[str, RelativeDirDTO] = {}
        self.registered_dataset: Dict[str, ChunksDTO] = {}
        self.new_dataset: Dict[str, ChunksDTO] = {}

    def fill_files_info(self):
        for filepath in scan_dir(self.source_dir, recursive=True):
            fpath = Path(filepath)
            relative_dir = fpath.relative_to(self.source_dir).parent
            file_info = FileDTO(
                filename=fpath.name,
                size=os.path.getsize(fpath),
                hash=str(sha256(open(fpath, "rb").read()).hexdigest()),
            )

            if self.dataset.get(str(relative_dir)) is None:
                self.dataset[str(relative_dir)] = RelativeDirDTO(dir=relative_dir)

            self.dataset[str(relative_dir)].files[str(file_info.filename)] = file_info

    def fill_registered_chunks(self, df: pd.DataFrame, relative_dir: Path):
        if dir_item := self.dataset.get(str(relative_dir)):
            self.registered_dataset[str(relative_dir)] = ChunksDTO(
                df=df, dir=relative_dir
            )

            for archive in set(df["archive"].tolist()):
                df_chunk = df[df["archive"] == archive]

                for i, row in df_chunk[
                    df_chunk["filename"].apply(lambda x: x in dir_item.files)
                ].iterrows():
                    if (file_info := dir_item.files.get(row["filename"])) and file_info.size == row["size"] and file_info.hash == row["hash"]:
                        file_info = dir_item.files.pop(row["filename"])
                        file_info.archive = row["archive"]
                        self.registered_dataset[str(relative_dir)].archives[
                            archive
                        ].append(file_info)

    def chunking(self, chunk_limit: int):
        for dir_path, dir_info in self.dataset.items():
            if not dir_info.files:
                continue

            batch_size = 0
            archive_name = f"{str(ULID())}.zip"
            self.new_dataset[dir_path] = ChunksDTO(dir=dir_info.dir)

            for filename, file_info in dir_info.files.items():
                # Если пачка полна
                if (batch_size := batch_size + file_info.size) > chunk_limit:
                    archive_name = f"{str(ULID())}.zip"
                    batch_size = file_info.size

                file_info.archive = archive_name
                self.new_dataset[dir_path].archives[archive_name].append(file_info)

            self.new_dataset[dir_path].df = pd.DataFrame(
                [
                    i.model_dump()
                    for i in self.new_dataset[dir_path].archives[archive_name]
                ]
            )

    def chunk_packing(
        self, archive_path: str, archive_name: str, files_info: List[FileDTO]
    ):
        # Архивирование
        arch_path = self.source_dir / archive_path / archive_name

        with ZipFile(arch_path, "w") as zipf:
            for file_info in sorted(files_info, key=lambda x: x.filename):
                zipf.write(
                    self.source_dir / archive_path / file_info.filename,
                    file_info.filename,
                )

        # Удаление исходных файлов
        for file_info in files_info:
            os.remove(self.source_dir / archive_path / file_info.filename)

    def index_dump(self):
        all_dirs = set(
            [str(i.dir) for _, i in self.registered_dataset.items()]
            + [str(i.dir) for _, i in self.new_dataset.items()]
        )
        for dir_path in all_dirs:
            df = None
            reg = self.registered_dataset.get(dir_path)
            new = self.new_dataset.get(dir_path)

            for item in [reg, new]:
                if item:
                    if df is None:
                        df = item.df
                    else:
                        df = pd.concat([df, item.df], ignore_index=True)

            df.to_csv(
                self.source_dir / dir_path / INDEX_FILENAME,
                sep=",",
                encoding="utf-8",
                index=False,
            )

    # ---------------------------------------------------------------------------------
    # -- ТОЧКИ ВХОДА ------------------------------------------------------------------
    # ---------------------------------------------------------------------------------
    def run(self, chunk_limit: int = 10 * 1024**2):
        logger.info("scanning files...")
        self.fill_files_info()

        # Находим неизмененные ранее зарегистрированные файлы
        logger.info("search registered files...")
        for index_path in find_indexes(self.source_dir):
            df = pd.read_csv(index_path, sep=",", encoding="utf-8")
            relative_dir = index_path.relative_to(self.source_dir).parent
            self.fill_registered_chunks(df, relative_dir)

        # Упаковываем в те же чанки неизмененные ранее зарегистрированные файлы
        if n := sum([len(i.archives) for _, i in self.registered_dataset.items()]):
            progress_bar = tqdm(total=n, desc="packing registered chunks")
            for dir_path, dir_info in self.registered_dataset.items():
                for archive_name, files_info in dir_info.archives.items():
                    self.chunk_packing(dir_path, archive_name, files_info)
                    progress_bar.update()
            progress_bar.close()
        else:
            logger.info("registered files not found")

        # Распределяем оставшиеся (новые) файлы по чанкам и упаковываем
        if n := sum([len(i.archives) for _, i in self.new_dataset.items()]):
            self.chunking(chunk_limit)
            progress_bar = tqdm(total=n, desc="packing new chunks")
            for dir_path, dir_info in self.new_dataset.items():
                for archive_name, files_info in dir_info.archives.items():
                    self.chunk_packing(dir_path, archive_name, files_info)
                    progress_bar.update()
            progress_bar.close()
        else:
            logger.info("new files not found")

        # Объединение и сохранение индексов
        self.index_dump()
        logger.info("done")


@cli_packing.command()
@click.option("--source-dir", "-s", help="Path to captcha directory")
def packing(source_dir: str):
    source_dir = Path(source_dir)
    DataPacker(source_dir).run()


__all__ = ["cli_packing"]
