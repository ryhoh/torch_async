import pickle

import pandas as pd

import torch
import torch.nn as nn
from torch.nn.modules import Linear
import torch.optim as optim

from torchvision.models import vgg

from procedure import preprocess
from layers.static import *
from models.dense import SimpleMLP, FeatureClassifyMLP, FeatureClassifyMLPFrontVer


GPU_ENABLED = False


def write_record(records: dict, epoch: int):
    pd.DataFrame({
        'train_loss': records['train_loss'],
        'train_accuracy': records['train_accuracy'],
    }).to_csv("semisync_" + str(epoch) + "_train.csv")

    pd.DataFrame({
        'validation_loss': records['validation_loss'],
        'validation_accuracy': records['validation_accuracy'],
    }).to_csv("semisync_" + str(epoch) + "_valid.csv")


def conduct(model: nn.Module, train_loader, test_loader) -> dict:
    def rotate_all():
        # try:
        #     layers = model.classifier
        # except AttributeError:
        #     layers = model
        #
        # for layer in layers:
        #     if isinstance(layer, Rotatable):
        #         layer.rotate()
        #
        # assert list(loss_vector.size()) == [mini_batch_size]

        try:
            for layer in model.features:
                if isinstance(layer, Rotatable):
                    layer.rotate()
        except AttributeError:
            pass

    def post_epoch():
        pass

    records = {
        'train_loss': [],
        'validation_loss': [],
        'train_accuracy': [],
        'validation_accuracy': [],
    }

    loss_layer = nn.CrossEntropyLoss(reduction='none')
    loss_layer_reduce = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    data_n = len(train_loader.dataset)

    # validation of pre_training
    validate(model, test_loader, records)

    # training
    for epoch in range(100):
        total_loss = 0.0
        total_correct = 0

        for i, mini_batch in enumerate(train_loader):
            input_data, label_data = mini_batch
            mini_batch_size = list(input_data.size())[0]

            if GPU_ENABLED:
                in_tensor = input_data.to('cuda')
                label_tensor = label_data.to('cuda')
            else:
                in_tensor = input_data
                label_tensor = label_data

            if i == 0:
                print("epoch started")
                print("mini_batch_size:", mini_batch_size)

            optimizer.zero_grad()  # Optimizer を0で初期化

            # forward - backward - optimize
            outputs = model(in_tensor)
            loss_vector = loss_layer(outputs, label_tensor)  # for evaluation
            reduced_loss = loss_layer_reduce(outputs, label_tensor)  # for backward
            _, predicted = torch.max(outputs.data, 1)

            reduced_loss.backward()
            optimizer.step()

            rotate_all()

            total_loss += loss_vector.data.sum().item()
            total_correct += (predicted.to('cpu') == label_data).sum().item()

            if i % 2000 == 1999:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, reduced_loss.item()))
                # running_loss = 0.0

        # end of epoch
        records['train_loss'].append(total_loss / data_n)
        records['train_accuracy'].append(total_correct / data_n)
        validate(model, test_loader, records)
        save(model)

        # rotate_all()

        if epoch % 20 == 0:
            write_record(records, epoch)

    print('Finished Training')
    return records


def validate(model: nn.Module, test_loader, records: dict):
    with torch.no_grad():
        loss_layer = nn.CrossEntropyLoss(reduction='none')

        total_correct = 0
        total_loss = 0.0
        data_n = len(test_loader.dataset)

        for mini_batch in test_loader:
            input_data, label_data = mini_batch
            mini_batch_size = list(input_data.size())[0]

            if GPU_ENABLED:
                in_tensor = input_data.to('cuda')
                label_tensor = label_data.to('cuda')
            else:
                in_tensor = input_data
                label_tensor = label_data

            outputs = model(in_tensor)
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


def save(model: nn.Module):
    with open("models.pkl", 'wb') as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    for semisync in (True,):
        torch.manual_seed(0)

        model = vgg.vgg16()
        # model = FeatureClassifyMLPFrontVer()
        #
        # if semisync:
        #     for layer_i in (0, 3):
        #         models.classifier[layer_i] = SemiSyncLinear(models.classifier[layer_i])
        #
        # if GPU_ENABLED:
        #     models.to('cuda')
        #
        if semisync:
        #     targets = [0]
        #     for target in targets:
        #         model.classifier[target] = OptimizedContinuousLinear(model.classifier[target])

            model.classifier[0] = OptimizedSemiSyncLinear(model.classifier[0])
            model.classifier[3] = OptimizedSemiSyncLinear(model.classifier[3])

        print(model)

        # model.train()
        # record = conduct(model, *(preprocess.mnist_loaders()))
        #
        # pd.DataFrame({
        #     'train_loss':     record['train_loss'],
        #     'train_accuracy': record['train_accuracy'],
        # }).to_csv("semisync_" + str(semisync) + "_train.csv")
        #
        # pd.DataFrame({
        #     'validation_loss':     record['validation_loss'],
        #     'validation_accuracy': record['validation_accuracy'],
        # }).to_csv("semisync_" + str(semisync) + "_valid.csv")
