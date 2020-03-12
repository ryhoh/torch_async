from typing import Tuple
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import os
import random
import numpy as np


class IMAGENET(Dataset):
    """ImageNet用オリジナルDataset

    ファイルパスカラム(path)とラベルカラム(label)を持つcsvを入力とし、
    Datasetをつくる

    Attributes:
        normalize (Normalize): 正規化?
        transform (transform): 前処理?
        _df (DataFrame): Dataset情報
        root_dir (str): データセットのルートディレクトリ
        images (dir): 画像の一覧

    Args:
        root_dir (str): データセットのルートディレクトリ
        csv_path (str): ファイルパスカラム(path)とラベルカラム(label)を持つcsv

    Note:
        水増しやrandomcropとrandomflipの前処理なし
    """
    def __init__(self, root_dir: str, csv_path: str):
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            self.normalize,
        ])
        self._df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        # csvから画像一覧を読み出す
        self.images = self._df['path'].values
        self.labels = self._df['label'].values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        image_name = self.images[idx]
        image = Image.open(os.path.join(self.root_dir, image_name))
        image = image.convert("RGB")
        if self.transforms:
            out_data = self.transforms(image)
        label = self.labels[idx]
        return out_data, int(label), image_name


seed = 0


def worker_init_fn(worker_id):
    """ DataLoaderでランダムシードを固定するための設定 """
    global seed

    random.seed(worker_id+seed)
    np.random.seed(worker_id+seed)


def imagenet_n_dataloaders(
        root_dir: str, train_csv_path: str, val_csv_path: str,
        a_csv_path: str, random_seed: int, batch_size: int
        ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """ ImageNetで評価セットが2つある時用のdataloader

    Notes:
        train時にDataSet内でrandomcropとrandomflipの前処理なし
    """
    global seed

    train_set = IMAGENET(root_dir, train_csv_path)
    test_set = IMAGENET(root_dir, val_csv_path)
    a_set = IMAGENET(root_dir, a_csv_path)

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
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=worker_init_fn)
        a_loader = DataLoader(
            a_set,
            batch_size=batch_size,
            shuffle=True,
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
            shuffle=True,
            num_workers=4,
            pin_memory=True)
        a_loader = DataLoader(
            a_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True)

    return train_loader, test_loader, a_loader


def imagenet_dataloaders(
        root_dir: str, train_csv_path: str, val_csv_path: str,
        random_seed: int, batch_size: int
        ) -> Tuple[DataLoader, DataLoader]:
    """ ImageNet用DataLoader

    ImageNet用のDataLoader

    Args:
        root_dir (str): データセットのルートディレクトリ
        train_csv_path (str): ファイルパスカラム(path)とラベルカラム(label)を持つcsvのパス
        val_csv_path (str): ファイルパスカラム(path)とラベルカラム(label)を持つcsvのパス

    Returns:
        Tuple[DataLoader, DataLoader]: dataloaderを返す

    Notes:
        train時にDataSet内でrandomcropとrandomflipの前処理なし
    """
    global seed

    train_set = IMAGENET(root_dir, train_csv_path)
    test_set = IMAGENET(root_dir, val_csv_path)

    """ dataloaderのパラメータ

    バッチサイズは先行研究と同じ値
    num_workers, pin_memoryはデータを並列に読むための設定
    worker_init_fn 再現性を保つためにランダムシードを固定
    """
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
            shuffle=True,
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
            shuffle=True,
            num_workers=4,
            pin_memory=True)

    return train_loader, test_loader


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
            shuffle=True,
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
            shuffle=True,
            num_workers=4,
            pin_memory=True)

    return train_loader, test_loader


def cifar100_dataloader(random_seed: int, batch_size: int) -> Tuple[DataLoader, DataLoader]:
    global seed
    # テンソル化, RGB毎に平均と標準偏差を用いて正規化
    transform = transforms.Compose([
        # VGGは元々ImageNetを想定しているので、cifarをリサイズする
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # data_sets
    train_set = CIFAR100(root='./data', train=True,
                         download=True, transform=transform)
    test_set = CIFAR100(root='./data', train=False,
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
            shuffle=True,
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
            shuffle=True,
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
