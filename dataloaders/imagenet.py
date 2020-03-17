from typing import Tuple
from torch.utils.data import DataLoader
from dataloaders.dataset import IMAGENET
import random
import numpy as np

seed = 0


def worker_init_fn(worker_id):
    """ DataLoaderでランダムシードを固定するための設定 """
    global seed

    random.seed(worker_id+seed)
    np.random.seed(worker_id+seed)


def imagenet_train_eval_eval_dataloaders(
        root_dir: str, train_csv_path: str, val_csv_path: str,
        a_csv_path: str, random_seed: int, batch_size: int, resolution: int
        ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """ ImageNetで評価セットが2つある時用のdataloader

    Notes:
        train時にDataSet内でrandomcropとrandomflipの前処理あり
    """
    global seed

    train_set = IMAGENET(root_dir, train_csv_path, "train", resolution)
    test_set = IMAGENET(root_dir, val_csv_path, "eval", resolution)
    a_set = IMAGENET(root_dir, a_csv_path, "eval", resolution)

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
        a_loader = DataLoader(
            a_set,
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
        a_loader = DataLoader(
            a_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True)

    return train_loader, test_loader, a_loader


def imagenet_train_eval_dataloaders(
        root_dir: str, train_csv_path: str, val_csv_path: str,
        random_seed: int, batch_size: int, resolution: int
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
        train時にDataSet内でrandomcropとrandomflipの前処理あり
    """
    global seed

    train_set = IMAGENET(root_dir, train_csv_path, "train", resolution)
    test_set = IMAGENET(root_dir, val_csv_path, "eval", resolution)

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
