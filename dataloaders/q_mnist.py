from typing import Tuple
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import QMNIST
import random
import numpy as np

seed = 0


def worker_init_fn(worker_id):
    """ DataLoaderでランダムシードを固定するための設定 """
    global seed

    random.seed(worker_id+seed)
    np.random.seed(worker_id+seed)


def q_mnist_loaders(
        random_seed: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    global seed

    # data_sets
    train_set = QMNIST(root='./data', train=True,
                       download=True, transform=transform)
    test_set = QMNIST(root='./data', train=False,
                      download=True, transform=transform)

    # data_loader
    if random_seed:
        seed = random_seed
        train_loader = DataLoader(
                        train_set,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True,
                        worker_init_fn=worker_init_fn)
        test_loader = DataLoader(
                        test_set,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True,
                        worker_init_fn=worker_init_fn)
    else:
        train_loader = DataLoader(
                        train_set,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)
        test_loader = DataLoader(
                        test_set,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True)

    return train_loader, test_loader
