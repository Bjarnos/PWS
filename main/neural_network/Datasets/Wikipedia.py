from typing import ClassVar
import os
import jax.numpy as np
import tiktoken
import numpy

from .Dataset import Dataset

class Wikipedia(Dataset):
    """
    The class for the Wikipedia dataset.

    It contains 60.000 training images+labels and
    10.000 test images+labels, each representing a number.
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
