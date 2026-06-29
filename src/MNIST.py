import requests
import os
import gzip
import numpy as np

class MNIST:
    def __init__(self, data_dir="data", base_url="https://ossci-datasets.s3.amazonaws.com/mnist/"):
        self.data_sources = {
            "training_images": "train-images-idx3-ubyte.gz",  # 60,000 training images.
            "test_images": "t10k-images-idx3-ubyte.gz",  # 10,000 test images.
            "training_labels": "train-labels-idx1-ubyte.gz",  # 60,000 training labels.
            "test_labels": "t10k-labels-idx1-ubyte.gz",  # 10,000 test labels.
        }

        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        self.base_url = base_url

        for fname in self.data_sources.values():
            fpath = os.path.join(self.data_dir, fname)
            if not os.path.exists(fpath):
                self.download(fname)

        self.decompress()
        
        # normalize data
        self.train_images = self.train_images / 255.0
        self.test_images  = self.test_images  / 255.0

        # one hot encode
        self.train_labels = (self.train_labels[..., None] == np.arange(10)[None]).astype(np.float64)
        self.test_labels  = (self.test_labels[..., None]  == np.arange(10)[None]).astype(np.float64)

    def get_train_images(self):
        return self.train_images
    
    def get_test_images(self):
        return self.test_images
    
    def get_train_labels(self):
        return self.train_labels
    
    def get_test_labels(self):
        return self.test_labels

    def download(self, filename):
        print("downloading dataset file: " + filename)

        filepath = os.path.join(self.data_dir, filename)
        resp = requests.get(self.base_url + filename, stream=True)
        resp.raise_for_status()
        with open(filepath, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=128):
                fh.write(chunk)

    def decompress(self):
        # images
        with gzip.open(os.path.join(self.data_dir, self.data_sources["training_images"]), "rb") as mnist_file:
            self.train_images = np.frombuffer(mnist_file.read(), np.uint8, offset=16).reshape(-1, 28 * 28).copy()
        with gzip.open(os.path.join(self.data_dir, self.data_sources["test_images"]), "rb") as mnist_file:
            self.test_images = np.frombuffer(mnist_file.read(), np.uint8, offset=16).reshape(-1, 28 * 28).copy()

        # labels
        with gzip.open(os.path.join(self.data_dir, self.data_sources["training_labels"]), "rb") as mnist_file:
            self.train_labels = np.frombuffer(mnist_file.read(), np.uint8, offset=8).copy()
        with gzip.open(os.path.join(self.data_dir, self.data_sources["test_labels"]), "rb") as mnist_file:
            self.test_labels = np.frombuffer(mnist_file.read(), np.uint8, offset=8).copy()

    def get_input_size(self):
        return len(self.train_images[0])
    
    def get_output_size(self):
        return len(self.train_labels[0])
