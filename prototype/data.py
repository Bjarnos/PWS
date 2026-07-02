import torch

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

from config import BATCH_SIZE


def create_datasets():
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    return training_data, test_data


def create_dataloaders(batch_size=BATCH_SIZE):
    training_data, test_data = create_datasets()

    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
    )

    return (
        training_data,
        test_data,
        train_dataloader,
        test_dataloader,
    )