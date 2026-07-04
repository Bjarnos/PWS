from typing import Optional, ClassVar
import os
import shutil
import kagglehub # pyright: ignore[reportMissingTypeStubs]

class Dataset:
    """
    The parent class for all datasets.

    <em>You should not use this class directly, but rather
    one of the child classes. It is only exported to use
    for typing.</em>
    """
    
    default_data_dir: ClassVar[str] = ""
    default_kaggle_name: ClassVar[str] = ""
    data_sources: ClassVar[dict[str, str]] = {}

    def __init__(self, data_dir: Optional[str] = None, kaggle_name: Optional[str] = None):
        """
        You can set a custom `data_dir`, which is where the downloaded
        files will be stored, and a custom `kaggle_name`, which is the
        [kaggle](https://www.kaggle.com/datasets) repository where the
        files are downloaded from.
        """

        if not data_dir:
            data_dir = self.default_data_dir
        
        if not kaggle_name:
            kaggle_name = self.default_kaggle_name

        os.makedirs(data_dir, exist_ok=True)

        should_download = False
        for fname in self.data_sources.values():
            fpath = os.path.join(data_dir, fname)
            if not os.path.exists(fpath):
                should_download = True

        if should_download:
            if os.path.exists(data_dir):
                shutil.rmtree(data_dir)
                os.makedirs(data_dir)
            
            print(kaggle_name)
            kagglehub.dataset_download(kaggle_name, output_dir=data_dir)

        self._setup(data_dir)

    def _setup(self, data_dir: str): pass
