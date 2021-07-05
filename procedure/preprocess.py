from typing import Tuple

from PIL import Image
from pycocotools.mask import frPyObjects, decode
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, CocoDetection
from torchvision import transforms


def cifar_100_for_224s() -> Tuple[DataLoader, DataLoader]:
    # テンソル化, RGB毎に平均と標準偏差を用いて正規化
    transform = transforms.Compose([
        # VGGは元々ImageNetを想定しているので、cifarをリサイズする
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # data_sets
    train_set = CIFAR100(root='~/dataset', train=True,
                        download=True, transform=transform)
    test_set = CIFAR100(root='~/dataset', train=False,
                       download=True, transform=transform)

    # data_loader
    train_loader = DataLoader(train_set, batch_size=32,
                              shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=32,
                             shuffle=False, num_workers=4)

    return train_loader, test_loader


def cifar_10_for_vgg_loaders() -> Tuple[DataLoader, DataLoader]:
    return cifar_10_for_224s()


def cifar_10_for_224s() -> Tuple[DataLoader, DataLoader]:
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


class CocoSegmentation(CocoDetection):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        for category in target:
            seg_rle = category['segmentation']
            category['segmentation'] = decode(frPyObjects(seg_rle, img.shape[1], img.shape[2]))
        return img, target


def CocoDetection_loaders() -> Tuple[DataLoader, DataLoader]:
    # テンソル化, RGB毎に平均と標準偏差を用いて正規化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # data_sets
    train_set = CocoDetection(root='~/dataset/coco/train2017',
                              annFile='~/dataset/coco/annotations/instances_train2017.json',
                              transform=transform)
    val_set = CocoDetection(root='~/dataset/coco/val2017',
                            annFile='~/dataset/coco/annotations/instances_val2017.json',
                            transform=transform)

    # data_loader
    train_loader = DataLoader(train_set, batch_size=32,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32,
                            shuffle=False, num_workers=4)

    return train_loader, val_loader
