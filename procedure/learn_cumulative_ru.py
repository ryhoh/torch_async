import sys
from typing import Dict, Any, Tuple, List

import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules import Linear, ReLU, Dropout
from torch.utils.data import DataLoader
from awesome_progress_bar import ProgressBar

from procedure import preprocess
from models import resnet110


lr = 0.001
momentum = 0.9


class StaggeredUpdateController(object):
    r"""
    1レイヤの更新をコントロールする
    長さ1の配列を渡せば従来法と同じ動作，それより大きい配列で Staggered する
    """
    # todo optimizer を使わずに自前で更新  c.f. https://qiita.com/perrying/items/857df46bb6cdc3047bd8#optimを使わない場合
    def __init__(self, optimizers: List[torch.optim.Optimizer]):
        self.optimizers = optimizers
        self.idx = 0

    def step(self):
        self.optimizers[self.idx].step()
        self.optimizers[self.idx].zero_grad()

    def rotate(self):
        self.idx = self.idx + 1 if self.idx + 1 != len(self.optimizers) else 0


class Learner(object):
    def __init__(self, model, train_loader, test_loader,
                 feature_params: list, fc_mid_controllers: List[StaggeredUpdateController], fc_end_param,
                 staggered_update: bool, gpu_idx: int = None, detail: str = None):

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

        self.optimizers = {  # 重みごとに optimizer を使い分ける
            'normal_optimizers': normal_optimizers,
            'staggered_controllers': fc_mid_controllers,
        }
        self.dataset = {
            'train': train_loader,
            'test':  test_loader,
            'train_length': len(train_loader.dataset),
            'test_length':  len(test_loader.dataset),
        }

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
        res = self.learner['model'].__class__.__name__
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
            sys.stderr.write('Loss: {:.3f}, Accuracy: {:.3f}\n'.format(
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

        # 全結合中間層
        for controller in self.optimizers['staggered_controllers']:
            controller.step()
            controller.rotate()

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

    r"""
    params = list(model.linear.parameters())[0]  # generator 展開
    print(params.shape)  # torch.Size([10, 64]), 64x10型行列
    print(params[0, :].shape)  # torch.Size([64]), 特定ニューロンのパラメータのみ選択できる
    
    これじゃ non_leaf Tensor って言われるから，扱えない
    """

    seed = int(sys.argv[1])
    sys.stderr.write("seed = " + str(seed) + '\n')

    for case in ('none', 'dropout', 'rotate'):
        torch.manual_seed(seed)
        model = resnet110(use_global_average_pooling=True)
        train_loader, test_loader = preprocess.cifar10_loaders()

        if case == 'rotate' or case == 'dropout':
            fc_mid_1 = Linear(in_features=4096, out_features=1024, bias=True)
            fc_mid_2 = Linear(in_features=1024, out_features=1024, bias=True)
            fc_end = Linear(in_features=1024, out_features=10, bias=True)
            fc_end_param = fc_end.parameters()

            if case == 'rotate':
                model.linear = nn.Sequential(
                    fc_mid_1,
                    ReLU(inplace=True),
                    fc_mid_2,
                    ReLU(inplace=True),
                    fc_end,
                )
                mid_1_param = list(model.linear[0].parameters())[0]
                mid_2_param = list(model.linear[2].parameters())[0]
                g_size = int(1024 ** 0.5)  # sqrt(32)
                fc_mid_controllers = [
                    StaggeredUpdateController([
                        torch.optim.SGD(
                            list(mid_1_param[g_size * i: g_size * i + g_size, :]),
                            lr=lr, momentum=momentum
                        )
                        for i in range(g_size)
                    ]),
                    StaggeredUpdateController([
                        torch.optim.SGD(
                            list(mid_2_param[g_size * i: g_size * i + g_size, :]),
                            lr=lr, momentum=momentum
                        )
                        for i in range(g_size)
                    ]),
                ]
            else:  # dropout
                model.linear = nn.Sequential(
                    fc_mid_1,
                    ReLU(inplace=True),
                    Dropout(0.5),
                    fc_mid_2,
                    ReLU(inplace=True),
                    Dropout(0.5),
                    fc_end,
                )
                fc_mid_controllers = [
                    StaggeredUpdateController([
                        torch.optim.SGD(model.linear[0].parameters(), lr=lr, momentum=momentum)
                    ]),
                    StaggeredUpdateController([
                        torch.optim.SGD(model.linear[3].parameters(), lr=lr, momentum=momentum)
                    ]),
                ]

        else:  # none
            model.linear = Linear(in_features=64, out_features=10, bias=True)
            fc_mid_controllers = []
            fc_end_param = model.linear.parameters()

        learner = Learner(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            feature_params=[
                model.layer1.parameters(),
                model.layer2.parameters(),
                model.layer3.parameters()
            ],
            fc_mid_controllers=fc_mid_controllers,
            fc_end_param=fc_end_param,
            staggered_update=(True if case == 'rotate' else False),
            detail=case
        )
        print(str(learner))

    # model = resnet110(use_global_average_pooling=True)
    # params = list(model.linear.parameters())[0]  # generator 展開
    # print(params.shape)  # torch.Size([10, 64]), 64x10型行列
    # print(params)
    # part_params = params[0, :]
    # print(part_params.shape)  # torch.Size([64]), 特定ニューロンのパラメータのみ選択できる
    # print(part_params)
    # print(part_params.is_leaf)
