from typing import Any, Dict, List
import os
from kedro.io import AbstractDataset, DatasetError
from ebooklib import epub


class EPUBDataSet(AbstractDataset):
    def __init__(self, filepath: str):
        self._filepath = filepath

    def _load(self) -> epub.EpubBook:
        if not os.path.exists(self._filepath):
            raise DatasetError(f"Filepath {self._filepath} does not exist.")

        book = epub.read_epub(self._filepath)
        return book

    def _save(self, data: Any) -> None:
        raise DatasetError("Saving data is not supported for EPUBDataSet")

    def _describe(self) -> Dict[str, Any]:
        return {"filepath": self._filepath}
