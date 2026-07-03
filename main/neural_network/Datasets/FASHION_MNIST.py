from .MNIST import MNIST

class FASHION_MNIST(MNIST):
    """
    The class for the Fashion-MNIST dataset.

    It contains 60.000 training images+labels and
    10.000 test images+labels, each representing a
    piece of clothing.
    
    The Fashion-MNIST dataset inherits all members
    of the MNIST dataset
    """

    default_data_dir: str = "data/fashion-mnist"
    default_kaggle_name: str = "zalando-research/fashionmnist"

    data_sources: dict[str, str] = {
        "training_images": "train-images-idx3-ubyte",
        "test_images": "t10k-images-idx3-ubyte",
        "training_labels": "train-labels-idx1-ubyte",
        "test_labels": "t10k-labels-idx1-ubyte",
    }
