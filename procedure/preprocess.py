import os
from typing import Tuple

import numpy as np
from PIL import Image
from pycocotools.mask import frPyObjects, decode
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, CocoDetection
from torchvision import transforms

from procedure.lib.cocodataset import COCODataset


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


# # http://akasuku.blog.jp/archives/73817244.html
# class CocoSegmentation(CocoDetection):
#     def __getitem__(self, index):
#
#         # データ入力
#         coco = self.coco
#         img_id = self.ids[index]
#         ann_ids = coco.getAnnIds(imgIds=img_id)
#         target = coco.loadAnns(ann_ids)
#         path = coco.loadImgs(img_id)[0]['file_name']
#         img = Image.open(os.path.join(self.root, path)).convert('RGB')
#
#         # セグメンテーション情報のデコード
#         for category in target:
#             seg_rle = category['segmentation']
#             tmp = decode(frPyObjects(seg_rle, img.size[1], img.size[0]))
#             if tmp.ndim == 3:
#                 tmp = np.sum(tmp, axis=2, dtype=np.uint8)
#             category['segmentation'] = tmp
#
#         # data_transform
#         if self.transform is not None:
#             img = self.transform(img)
#
#         # target_transform
#         for category in target:
#             pilImg = Image.fromarray(category['segmentation'])
#             tmp = pilImg.resize((img.shape[2], img.shape[1]), resample=Image.NEAREST)
#             target_transform = transforms.Compose([
#                 transforms.ToTensor()
#             ])
#             category['segmentation'] = target_transform(tmp)
#
#         return img, target


def CocoDetection_2017_loaders() -> Tuple[DataLoader, DataLoader]:
    # テンソル化, RGB毎に平均と標準偏差を用いて正規化
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 画像ごとにサイズが異なる
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # data_sets
    train_set = COCODataset(
        model_type='YOLOv3',
        data_dir='~/dataset/coco/train2017',
        json_file='~/dataset/coco/annotations_2017/instances_train2017.json',
        name='train2017',
        img_size=416)
    val_set = COCODataset(
        model_type='YOLOv3',
        data_dir='~/dataset/coco/val2017',
        json_file='~/dataset/coco/annotations_2017/instances_val2017.json',
        name='val2017',
        img_size=416)

    # data_loader
    train_loader = DataLoader(train_set, batch_size=32,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32,
                            shuffle=False, num_workers=4)

    return train_loader, val_loader


# def CocoDetection_2014_loaders() -> Tuple[DataLoader, DataLoader]:
#     # テンソル化, RGB毎に平均と標準偏差を用いて正規化
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),  # 画像ごとにサイズが異なる
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#
#     # data_sets
#     train_set = CocoDetection(root='~/dataset/coco/train2014',
#                               annFile='../dataset/coco/annotations_2014/instances_train2014.json',  # なぜか ../ でないと落ちる
#                               transform=transform)
#     val_set = CocoDetection(root='~/dataset/coco/val2014',
#                             annFile='../dataset/coco/annotations_2014/instances_val2014.json',
#                             transform=transform)
#
#     # data_loader
#     train_loader = DataLoader(train_set, batch_size=32,
#                               shuffle=True, num_workers=4)
#     val_loader = DataLoader(val_set, batch_size=32,
#                             shuffle=False, num_workers=4)
#
#     return train_loader, val_loader
