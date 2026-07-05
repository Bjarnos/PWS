from typing import ClassVar
import os
import jax.numpy as np
import numpy

from .Dataset import Dataset

class MNIST(Dataset):
    """
    The class for the MNIST dataset.

    It contains 60.000 training images+labels and
    10.000 test images+labels, each representing a number.
    """
    
    default_data_dir: ClassVar[str] = "data/mnist"
    default_kaggle_name: ClassVar[str] = "hojjatk/mnist-dataset"

    data_sources: ClassVar[dict[str, str]] = {
        "training_images": "train-images-idx3-ubyte/train-images-idx3-ubyte",
        "test_images": "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte",
        "training_labels": "train-labels-idx1-ubyte/train-labels-idx1-ubyte",
        "test_labels": "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte",
    }

    train_images: numpy.ndarray
    train_labels: numpy.ndarray
    test_images: numpy.ndarray
    test_labels: numpy.ndarray

    def _setup(self, data_dir: str):
        # decompress images
        with open(os.path.join(data_dir, self.data_sources["training_images"]), "rb") as mnist_file:
            self.train_images = numpy.frombuffer(mnist_file.read(), numpy.uint8, offset=16).reshape(-1, 28 * 28).copy()
        with open(os.path.join(data_dir, self.data_sources["test_images"]), "rb") as mnist_file:
            self.test_images = numpy.frombuffer(mnist_file.read(), numpy.uint8, offset=16).reshape(-1, 28 * 28).copy()

        # decompress labels
        with open(os.path.join(data_dir, self.data_sources["training_labels"]), "rb") as mnist_file:
            self.train_labels = numpy.frombuffer(mnist_file.read(), numpy.uint8, offset=8).copy()
        with open(os.path.join(data_dir, self.data_sources["test_labels"]), "rb") as mnist_file:
            self.test_labels = numpy.frombuffer(mnist_file.read(), numpy.uint8, offset=8).copy()
        
        # normalize data
        self.train_images = self.train_images / 255.0
        self.test_images  = self.test_images  / 255.0

        # one hot encode
        self.train_labels = (self.train_labels[..., None] == numpy.arange(10)[None]).astype(np.float32)
        self.test_labels  = (self.test_labels[..., None]  == numpy.arange(10)[None]).astype(np.float32)

        self.train_images = np.asarray(self.train_images) # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        self.train_labels = np.asarray(self.train_labels) # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        self.test_images = np.asarray(self.test_images) # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        self.test_labels = np.asarray(self.test_labels) # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]

    def get_input_size(self):
        "Returns the required input_size of the first layer."
        return len(self.train_images[0])
    
    def get_output_size(self):
        "Returns the required output_size of the final layer."
        return len(self.train_labels[0])
