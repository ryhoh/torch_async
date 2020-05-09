import sys
from typing import Tuple

import pandas as pd

import torch
import torch.nn as nn
from torch.nn.modules import Linear
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import vgg

from procedure import preprocess
from layers import SemisyncLinear, SequentialLinear, Rotatable


GPU_ENABLED = True
device = "cuda:1"


def rotate_all(learner: dict):
    for layer in learner['model'].classifier:
        if isinstance(layer, Rotatable):
            layer.rotate()


def conduct(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader) -> dict:
    records = {
        'train_loss': [],
        'validation_loss': [],
        'train_accuracy': [],
        'validation_accuracy': [],
    }
    learner = {
        'model': model,
        'loss_layer_reduce': nn.CrossEntropyLoss(),
        'loss_layer': nn.CrossEntropyLoss(reduction='none'),
        'optimizer': optim.SGD(model.parameters(), lr=0.001, momentum=0),
    }
    dataset = {
        'train': train_loader,
        'test': test_loader,
        'length': len(train_loader.dataset),
    }

    # 全く学習していない状態で測定
    records = validate(learner, dataset, records)

    # training
    for epoch in range(100):
        learner, records = run_epoch(learner, dataset['train'], dataset['length'], records, epoch)

        # 測定
        records = validate(learner, dataset, records)

    print('Finished Training')
    return records


def run_epoch(learner: dict, train_loader: DataLoader, data_n: int,
              records: dict, epoch: int) -> Tuple[dict, dict]:
    learner['model'].train()
    total_loss = 0.0
    total_correct = 0

    for i, mini_batch in enumerate(train_loader):
        input_data, label_data = mini_batch
        # mini_batch_size = list(input_data.size())[0]

        if GPU_ENABLED:
            in_tensor = input_data.to(device)
            label_tensor = label_data.to(device)
        else:
            in_tensor = input_data
            label_tensor = label_data

        if i == 0:
            print("epoch%04d started" % epoch)

        learner['optimizer'].zero_grad()  # Optimizer を0で初期化

        # forward - backward - optimize
        outputs = learner['model'](in_tensor)
        loss_vector = learner['loss_layer'](outputs, label_tensor)  # for evaluation
        reduced_loss = learner['loss_layer_reduce'](outputs, label_tensor)  # for backward
        _, predicted = torch.max(outputs.data, 1)

        reduced_loss.backward()
        learner['optimizer'].step()

        total_loss += loss_vector.data.sum().item()
        total_correct += (predicted.to('cpu') == label_data).sum().item()

        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, reduced_loss.item()))

        # 準同期式のグループ交代
        rotate_all(learner)

    records['train_loss'].append(total_loss / data_n)
    records['train_accuracy'].append(total_correct / data_n)

    return learner, records


def validate(learner: dict, dataset: dict, records: dict) -> dict:
    with torch.no_grad():
        loss_layer = nn.CrossEntropyLoss(reduction='none')

        total_correct = 0
        total_loss = 0.0
        data_n = len(dataset['test'].dataset)

        for mini_batch in dataset['test']:
            input_data, label_data = mini_batch
            mini_batch_size = list(input_data.size())[0]

            if GPU_ENABLED:
                in_tensor = input_data.to(device)
                label_tensor = label_data.to(device)
            else:
                in_tensor = input_data
                label_tensor = label_data

            outputs = learner['model'](in_tensor)
            loss_vector = loss_layer(outputs, label_tensor)
            _, predicted = torch.max(outputs.data, 1)

            assert list(loss_vector.size()) == [mini_batch_size]

            total_correct += (predicted.to('cpu') == label_data).sum().item()
            total_loss += loss_vector.sum().item()

        accuracy = total_correct / data_n
        loss_per_record = total_loss / data_n
        print('Loss: {:.3f}, Accuracy: {:.3f}'.format(
            loss_per_record,
            accuracy
        ))
        records['validation_loss'].append(loss_per_record)
        records['validation_accuracy'].append(accuracy)
    
    return records


def write_final_record(records: dict, group_size: int) -> None:
    case = {4096: "Linear", 64: "Semisync", 1: "Sequential"}

    pd.DataFrame({
        'train_loss': records['train_loss'],
        'train_accuracy': records['train_accuracy'],
    }).to_csv(case[group_size] + "_train.csv")

    pd.DataFrame({
        'validation_loss': records['validation_loss'],
        'validation_accuracy': records['validation_accuracy'],
    }).to_csv(case[group_size] + "_valid.csv")


def main(seed: int):
    print("seed =", seed)

    for N in (1, 64, 4096):
        torch.manual_seed(seed)
        myvgg = vgg.vgg16()

        assert isinstance(myvgg.classifier[0], Linear)
        assert isinstance(myvgg.classifier[3], Linear)
        if N == 1:  # 逐次
            myvgg.classifier[0] = SequentialLinear(myvgg.classifier[0])
            myvgg.classifier[3] = SequentialLinear(myvgg.classifier[3])
        elif N == 64:  # 準同期
            myvgg.classifier[0] = SemisyncLinear(myvgg.classifier[0])
            myvgg.classifier[3] = SemisyncLinear(myvgg.classifier[3])

        myvgg.classifier[-1] = Linear(4096, 10)

        # Dropout 抜き
        myvgg.classifier = nn.Sequential(
            myvgg.classifier[0],  # Linear  (Semi)
            myvgg.classifier[1],  # ReLU
            myvgg.classifier[3],  # Linear  (Semi)
            myvgg.classifier[4],  # ReLU
            myvgg.classifier[6],  # Linear
        )

        print(myvgg)
        myvgg.to(device)
        record = conduct(myvgg, *(preprocess.cifar_10_for_vgg_loaders()))
        write_final_record(record, N)


if __name__ == '__main__':
    seed = int(sys.argv[1])
    main(seed)
