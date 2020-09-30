import sys
from typing import Dict, Any

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from awesome_progress_bar import ProgressBar

from procedure import preprocess
from models import resnet110


class Learner(object):

    def __init__(self, model, train_loader, test_loader,
                 feature_params: list, fc_mid_params: list, fc_end_param,
                 staggered_update: bool, gpu_idx: int = None, detail: str = None):
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
            'loss_layer_accum': nn.CrossEntropyLoss(),
            'loss_layer_score':  nn.CrossEntropyLoss(reduction='none'),
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
            'test_length':  len(test_loader.dataset),
        }

        self.staggered_update = staggered_update
        self.staggered_idx = 0
        self.detail = detail

    def __repr__(self):
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

    def __str__(self):
        res = str(self.learner['model'].__class__.__name__),
        if self.detail:
            res += '_' + self.detail

        return res

    def run(self, epochs: int):
        # 全く学習していない状態で測定
        self.validate()

        for epoch in range(epochs):
            sys.stderr.write("epoch%04d started" % epoch)
            self.run_epoch()
            self.validate()

        self.write_final_record()

    def run_epoch(self):
        self.learner['model'].train()
        data_n = self.dataset['train_length']
        total_loss = 0.0
        total_correct = 0
        progressbar = ProgressBar(len(self.dataset['train_loader']))

        for i, mini_batch in enumerate(self.dataset['train_loader']):
            in_tensor, label_tensor = mini_batch
            in_tensor = in_tensor.to(self.device)
            label_tensor = label_tensor.to(self.device)

            forward_result = self._forward(
                self.learner['model'], in_tensor, label_tensor,
                [self.learner['loss_layer_accum'], [self.learner['loss_layer_score']]]
            )
            loss_vector = forward_result['loss'][1]  # for accumulated loss
            reduced_loss = forward_result['loss'][0]  # for score evaluation
            predicted = forward_result['predicted']

            reduced_loss.backward()
            self._optimize_step()

            total_correct += self._correct_sum(predicted, label_tensor)
            total_loss += self._loss_sum(loss_vector)

            progressbar.iter()

        progressbar.wait()
        self.records['train_loss'].append(total_loss / data_n)
        self.records['train_accuracy'].append(total_correct / data_n)

    def validate(self):
        self.learner['model'].eval()
        with torch.no_grad():
            total_correct = 0
            total_loss = 0.0
            data_n = self.dataset['test_length']

            for mini_batch in self.dataset['test']:
                in_tensor, label_tensor = mini_batch
                mini_batch_size = list(in_tensor.size())[0]

                in_tensor = in_tensor.to(self.device)
                label_tensor = label_tensor.to(self.device)

                forward_result = self._forward(
                    self.learner['model'], in_tensor, label_tensor, [self.learner['loss_layer_score']]
                )
                loss_vector = forward_result['loss'][0]
                predicted = forward_result['predicted']
                assert list(loss_vector.size()) == [mini_batch_size]

                total_correct += self._correct_sum(predicted, label_tensor)
                total_loss += self._loss_sum(loss_vector)

            accuracy = total_correct / data_n
            loss_per_record = total_loss / data_n
            sys.stderr.write('Loss: {:.3f}, Accuracy: {:.3f}'.format(
                loss_per_record,
                accuracy
            ))
            self.records['validation_loss'].append(loss_per_record)
            self.records['validation_accuracy'].append(accuracy)

    def write_final_record(self):
        pd.DataFrame({
            'train_loss':     self.records['train_loss'],
            'train_accuracy': self.records['train_accuracy'],
        }).to_csv(str(self) + "_train.csv")

        pd.DataFrame({
            'validation_loss':     self.records['validation_loss'],
            'validation_accuracy': self.records['validation_accuracy'],
        }).to_csv(str(self) + "_valid.csv")

    def _optimize_step(self):
        # staggered 非適応のパラメータ
        for optimizer in self.optimizers['normal_optimizers']:
            optimizer.step()
            optimizer.zero_grad()

        if self.staggered_update:
            # staggered 適応のパラメータ
            selected_optimizer = self.optimizers['staggered_optimizers'][self.staggered_idx]
            selected_optimizer.step()
            selected_optimizer.zero_grad()

            self.staggered_idx = self.staggered_idx + 1 \
                if self.staggered_idx != len(self.optimizers['staggered_optimizers']) else 0

        else:  # normal update
            for optimizer in self.optimizers['staggered_optimizers']:
                optimizer.step()
                optimizer.zero_grad()

    r"""
    評価用，勾配計算用の複数の誤差レイヤに対してフォワード
    """
    @staticmethod
    def _forward(model, in_tensor, label_tensor, loss_layers: list) -> Dict[str, Any]:
        outputs = model(in_tensor)
        _, predicted = torch.max(outputs.data, 1)

        return {
            'loss': tuple(loss_layer(outputs, label_tensor) for loss_layer in loss_layers),
            'predicted': predicted.to('cpu'),
        }

    @staticmethod
    def _correct_sum(predicted: torch.Tensor, label_tensor: torch.Tensor) -> float:
        return (predicted.to('cpu') == label_tensor.to('cpu')).sum().item()

    @staticmethod
    def _loss_sum(loss_vector):
        return loss_vector.sum().item()

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
