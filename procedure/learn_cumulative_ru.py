import sys
from typing import Tuple

import pandas as pd

import torch
import torch.nn as nn
from torch.nn.modules import Linear
import torch.optim as optim
from torch.utils.data import DataLoader
# from torchvision.models import vgg

from procedure import preprocess
from models import resnet110


class Learner(object):

    def __init__(self, model, train_loader, test_loader,
                 feature_params: list, fc_mid_params: list, fc_end_param,
                 gpu_idx: int = None):
        lr = 0.001
        momentum = 0.9

        if gpu_idx:
            self.GPU_ENABLED = True
            self.device = "cuda:" + str(gpu_idx)
        else:
            self.GPU_ENABLED = False
            self.device = "cpu"

        self.records = {
            'train_loss': [],
            'validation_loss': [],
            'train_accuracy': [],
            'validation_accuracy': [],
        }
        self.learner = {
            'model': model.to(self.device),
            'loss_layer_reduce': nn.CrossEntropyLoss(),
            'loss_layer': nn.CrossEntropyLoss(reduction='none'),
        }

        normal_optimizers = [
            optim.SGD(feature_param, lr=lr, momentum=momentum)
            for feature_param in feature_params]
        normal_optimizers.append(optim.SGD(fc_end_param, lr=lr, momentum=momentum))

        staggered_optimizers = [
            optim.SGD(fc_mid_param, lr=lr, momentum=momentum)
            for fc_mid_param in fc_mid_params]

        self.optimizers = {  # 重みごとに optimizer を使い分ける
            'normal_optimizers': normal_optimizers,
            'staggered_optimizers': staggered_optimizers,
        }
        self.dataset = {
            'train': train_loader,
            'test':  test_loader,
            'train_length': len(train_loader.dataset),
            'test_length':  len(train_loader.dataset),
        }

        self.repr_str = ', '.join([
            repr(model),
            repr(train_loader),
            repr(test_loader)
        ])

        if gpu_idx:
            self.repr_str += ', ' + repr(gpu_idx)

    def __repr__(self):
        return self.repr_str

    def __str__(self):
        return r"""
cumulative_rotational_update learner

records:
""" + str(self.records) +\
r"""

model:
""" + str(self.learner['model'].__class__.__name__) +\
r"""

optimizer:
""" + str(self.optimizers) +\
r"""

dataset:
""" + str(self.dataset) +\
r"""

device:
""" + str(self.device)




#     def conduct(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader) -> dict:
#         # 全く学習していない状態で測定
#         self.validate()
#
#         # training
#         for epoch in range(100):
#             learner, records = self.run_epoch(learner, dataset['train'], dataset['length'], records, epoch)
#
#             # 測定
#             records = validate(learner, dataset, records)
#
#         print('Finished Training')
#         return records
#
#
# def run_epoch(learner: dict, train_loader: DataLoader, data_n: int,
#               records: dict, epoch: int) -> Tuple[dict, dict]:
#     learner['model'].train()
#     total_loss = 0.0
#     total_correct = 0
#
#     for i, mini_batch in enumerate(train_loader):
#         input_data, label_data = mini_batch
#         # mini_batch_size = list(input_data.size())[0]
#
#         if GPU_ENABLED:
#             in_tensor = input_data.to(device)
#             label_tensor = label_data.to(device)
#         else:
#             in_tensor = input_data
#             label_tensor = label_data
#
#         if i == 0:
#             print("epoch%04d started" % epoch)
#
#         learner['optimizer'].zero_grad()  # Optimizer を0で初期化
#
#         # forward - backward - optimize
#         outputs = learner['model'](in_tensor)
#         loss_vector = learner['loss_layer'](outputs, label_tensor)  # for evaluation
#         reduced_loss = learner['loss_layer_reduce'](outputs, label_tensor)  # for backward
#         _, predicted = torch.max(outputs.data, 1)
#
#         reduced_loss.backward()
#         learner['optimizer'].step()
#
#         total_loss += loss_vector.data.sum().item()
#         total_correct += (predicted.to('cpu') == label_data).sum().item()
#
#         if i % 200 == 199:
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, reduced_loss.item()))
#
#         # 準同期式のグループ交代
#         rotate_all(learner)
#
#     records['train_loss'].append(total_loss / data_n)
#     records['train_accuracy'].append(total_correct / data_n)
#
#     return learner, records
#
#
#     def validate(self, learner: dict, dataset: dict, records: dict) -> dict:
#         with torch.no_grad():
#             loss_layer = nn.CrossEntropyLoss(reduction='none')
#
#             total_correct = 0
#             total_loss = 0.0
#             data_n = len(dataset['test'].dataset)
#
#             for mini_batch in dataset['test']:
#                 input_data, label_data = mini_batch
#                 mini_batch_size = list(input_data.size())[0]
#
#                 if GPU_ENABLED:
#                     in_tensor = input_data.to(device)
#                     label_tensor = label_data.to(device)
#                 else:
#                     in_tensor = input_data
#                     label_tensor = label_data
#
#                 outputs = learner['model'](in_tensor)
#                 loss_vector = loss_layer(outputs, label_tensor)
#                 _, predicted = torch.max(outputs.data, 1)
#
#                 assert list(loss_vector.size()) == [mini_batch_size]
#
#                 total_correct += (predicted.to('cpu') == label_data).sum().item()
#                 total_loss += loss_vector.sum().item()
#
#             accuracy = total_correct / data_n
#             loss_per_record = total_loss / data_n
#             print('Loss: {:.3f}, Accuracy: {:.3f}'.format(
#                 loss_per_record,
#                 accuracy
#             ))
#             records['validation_loss'].append(loss_per_record)
#             records['validation_accuracy'].append(accuracy)
#
#         return records
#
#
# def write_final_record(records: dict, group_size: int) -> None:
#     case = {4096: "Linear", 64: "Semisync", 1: "Sequential"}
#
#     pd.DataFrame({
#         'train_loss': records['train_loss'],
#         'train_accuracy': records['train_accuracy'],
#     }).to_csv(case[group_size] + "_train.csv")
#
#     pd.DataFrame({
#         'validation_loss': records['validation_loss'],
#         'validation_accuracy': records['validation_accuracy'],
#     }).to_csv(case[group_size] + "_valid.csv")
#
#
# def main(seed: int):
#     print("seed =", seed)
#
#     for N in (1, 64, 4096):
#         torch.manual_seed(seed)
#         myvgg = vgg.vgg16()
#
#         assert isinstance(myvgg.classifier[0], Linear)
#         assert isinstance(myvgg.classifier[3], Linear)
#         if N == 1:  # 逐次
#             myvgg.classifier[0] = SequentialLinear(myvgg.classifier[0])
#             myvgg.classifier[3] = SequentialLinear(myvgg.classifier[3])
#         elif N == 64:  # 準同期
#             myvgg.classifier[0] = SemisyncLinear(myvgg.classifier[0])
#             myvgg.classifier[3] = SemisyncLinear(myvgg.classifier[3])
#
#         myvgg.classifier[-1] = Linear(4096, 10)
#
#         # Dropout 抜き
#         myvgg.classifier = nn.Sequential(
#             myvgg.classifier[0],  # Linear  (Semi)
#             myvgg.classifier[1],  # ReLU
#             myvgg.classifier[3],  # Linear  (Semi)
#             myvgg.classifier[4],  # ReLU
#             myvgg.classifier[6],  # Linear
#         )
#
#         print(myvgg)
#         myvgg.to(device)
#         record = conduct(myvgg, *(preprocess.cifar_10_for_vgg_loaders()))
#         write_final_record(record, N)


if __name__ == '__main__':
    # seed = int(sys.argv[1])
    # main(seed)

    model = resnet110(use_global_average_pooling=True)
    # train_loader, test_loader = preprocess.cifar10_loaders()
    #
    # learner = Learner(
    #     model=model,
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     # feature_params=model.layer1.parameters(),
    #     feature_params=[
    #         model.layer1.parameters(),
    #         model.layer2.parameters(),
    #         model.layer3.parameters()
    #     ],
    #     fc_mid_params=[],
    #     fc_end_param=model.linear.parameters(),
    # )
    # print(str(learner))

    params = list(model.linear.parameters())[0]  # generator 展開
    print(params.shape)  # torch.Size([10, 64]), 64x10型行列
    print(params[0, :].shape)  # torch.Size([64]), 特定ニューロンのパラメータのみ選択できる
