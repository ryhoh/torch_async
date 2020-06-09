from typing import Tuple

from PIL import Image

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision import transforms
# ランダムシード
import random
import numpy as np

seed = 0


def worker_init_fn(worker_id):
    """ DataLoaderでランダムシードを固定するための設定 """
    global seed

    random.seed(worker_id+seed)
    np.random.seed(worker_id+seed)


def cifar10_dataloader(random_seed: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    global seed
    # テンソル化, RGB毎に平均と標準偏差を用いて正規化
    transform = transforms.Compose([
        # VGGは元々ImageNetを想定しているので、cifarをリサイズする
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # data_sets
    train_set = CIFAR10(root='./data', train=True,
                        download=True, transform=transform)
    test_set = CIFAR10(root='./data', train=False,
                       download=True, transform=transform)

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


def cifar_10_for_vgg_loaders() -> Tuple[DataLoader, DataLoader]:
    # テンソル化, RGB毎に平均と標準偏差を用いて正規化
    transform = transforms.Compose([
        # VGGは元々ImageNetを想定しているので、cifarをリサイズする
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # data_sets
    train_set = CIFAR10(root='./data', train=True,
                        download=True, transform=transform)
    test_set = CIFAR10(root='./data', train=False,
                       download=True, transform=transform)

    # data_loader
    train_loader = DataLoader(train_set, batch_size=32,
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=32,
                             shuffle=False, num_workers=4)

    return train_loader, test_loader


def cifar10_loaders() -> Tuple[DataLoader, DataLoader]:
    # テンソル化, RGB毎に平均と標準偏差を用いて正規化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # data_sets
    train_set = CIFAR10(root='./data', train=True,
                        download=True, transform=transform)
    test_set = CIFAR10(root='./data', train=False,
                       download=True, transform=transform)

    # data_loader
    train_loader = DataLoader(train_set, batch_size=32,
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=32,
                             shuffle=False, num_workers=4)

    return train_loader, test_loader


def mnist_loaders() -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # data_sets
    train_set = MNIST(root='./data', train=True,
                      download=True, transform=transform)
    test_set = MNIST(root='./data', train=False,
                     download=True, transform=transform)

    # data_loader
    train_loader = DataLoader(train_set, batch_size=4,
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=4,
                             shuffle=False, num_workers=4)

    return train_loader, test_loader


def fashion_mnist_train_test_loader() -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # data_sets
    train_set = FashionMNIST(root='./data', train=True,
                             download=True, transform=transform)
    test_set = FashionMNIST(root='./data', train=False,
                            download=True, transform=transform)

    # data_loader
    train_loader = DataLoader(train_set, batch_size=4,
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=4,
                             shuffle=False, num_workers=4)

    return train_loader, test_loader
