from typing import Tuple

from PIL import Image

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torchvision import transforms


def cifar_10_for_vgg_loaders() -> Tuple[DataLoader, DataLoader]:
    # テンソル化, RGB毎に平均と標準偏差を用いて正規化
    transform = transforms.Compose([
        # VGGは元々ImageNetを想定しているので、cifarをリサイズする
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # data_sets
    train_set = CIFAR10(root='~/dataset', train=True,
                        download=True, transform=transform)
    test_set = CIFAR10(root='~/dataset', train=False,
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
    train_set = CIFAR10(root='~/dataset', train=True,
                        download=True, transform=transform)
    test_set = CIFAR10(root='~/dataset', train=False,
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
