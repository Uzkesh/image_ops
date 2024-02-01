from pathlib import Path

from handlers.utils.readers import scan_dir

INDEX_FILENAME = "dvc_chunks.csv"


def find_indexes(dir_path: Path):
    for i in scan_dir(dir_path, extensions=".csv", recursive=True):
        index_path = Path(i)
        if str(index_path.name) == INDEX_FILENAME:
            yield index_path
