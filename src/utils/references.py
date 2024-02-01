IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp")
SKIP_DIRS = frozenset(
    {
        ".ipynb_checkpoints",
        ".pytest_cache",
        "__pycache__",
        ".vscode",
        ".venv",
        "venv",
        ".idea",
        ".git",
        ".dvc",
    }
)

__all__ = ["IMAGE_EXTENSIONS", "SKIP_DIRS"]
