from typing import Any, Dict, List
import os
from kedro.io import AbstractDataset, DatasetError
from ebooklib import epub


class EPUBDataSet(AbstractDataset):
    def __init__(self, filepath: str):
        self._filepath = filepath

    def _load(self) -> List[str]:
        if not os.path.exists(self._filepath):
            raise DatasetError(f"Filepath {self._filepath} does not exist.")

        book = epub.read_epub(self._filepath)
        item_ids = [s[0] for s in book.spine]
        xhtml_content = []

        for item_id in item_ids:
            if item_id != "titlepage":
                item = book.get_item_with_id(item_id)
                xhtml_content.append(item.get_content().decode("utf-8"))

        return xhtml_content

    def _save(self, data: Any) -> None:
        raise DatasetError("Saving data is not supported for EPUBDataSet")

    def _describe(self) -> Dict[str, Any]:
        return {"filepath": self._filepath}
