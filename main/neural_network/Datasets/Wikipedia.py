from typing import ClassVar
import os

from .Dataset import Dataset

class Wikipedia(Dataset):
    """
    The class for the Wikipedia dataset.

    It contains 249.396 articles, 196.000 words, 31M tokens,
    all from the Simple English Wikipedia.
    """
    
    default_data_dir: ClassVar[str] = "data/wikipedia"
    default_kaggle_name: ClassVar[str] = "ffatty/plain-text-wikipedia-simpleenglish"

    data_sources: ClassVar[dict[str, str]] = {
        "text_combined": "AllCombined.txt",
    }

    text: str

    def _setup(self, data_dir: str):
        with open(os.path.join(data_dir, self.data_sources["text_combined"]), "r", encoding="utf-8") as f:
            self.text = f.read()
