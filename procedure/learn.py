import argparse
import sys
from typing import Dict, Any

import pandas as pd

import torch
import torch.nn as nn
from torch.nn.modules import Linear, ReLU, Dropout
import torch.optim as optim
from awesome_progress_bar import ProgressBar
from rotational_update import RotationalLinear, Rotatable

from procedure import preprocess
from models import resnet110


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Learner(object):
    def __init__(self, model, train_loader, test_loader,
                 seed_str: str, gpu_idx: int = None, detail: str = None,
                 momentum: float = 0.9, lr: float = 0.001):

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

        self.optimizers = {
            'optimizers': optim.SGD(self.learner['model'].parameters(), lr=lr, momentum=momentum),
        }
        self.dataset = {
            'train': train_loader,
            'test':  test_loader,
            'train_length': len(train_loader.dataset),
            'test_length':  len(test_loader.dataset),
        }

        self.detail = detail
        self.seed_str = seed_str

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
        res = self.learner['model'].__class__.__name__ + "_" + str(self.seed_str)
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

        torch.save(self.learner['model'].state_dict(), str(self) + '.torchmodel')

    def _optimize_step(self):
        self.optimizers['optimizer'].step()

    def _rotate_all(self):
        if 'classifier' in dir(self.learner['model']):
            fcs = self.learner['model'].classifier
        elif 'linear' in dir(self.learner['model']):
            fcs = self.learner['model'].linear
        elif 'fc' in dir(self.learner['model']):
            fcs = self.learner['model'].fc
        else:
            raise AttributeError('model has no fc!')

        for layer in fcs:
            if isinstance(layer, Rotatable):
                layer.rotate()

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


def main(seed: int, gpu_idx: int):
    sys.stderr.write("seed = " + str(seed) + '\n')
    sys.stderr.write("gpu = " + str(gpu_idx) + '\n')

    for case in ('none', 'rotational', 'dropout'):
        torch.manual_seed(seed)
        mymodel = resnet110(True if case == 'none' else False)

        if case == 'none':
            mymodel.linear = Linear(in_features=64, out_features=10, bias=True)

        elif case == 'rotate' or case == 'dropout':
            fc_mid_1 = Linear(in_features=4096, out_features=1024, bias=True)
            fc_mid_2 = Linear(in_features=1024, out_features=1024, bias=True)
            fc_end = Linear(in_features=1024, out_features=10, bias=True)

            if case == 'rotate':
                mymodel.linear = nn.Sequential(
                    RotationalLinear(fc_mid_1),
                    ReLU(inplace=True),
                    RotationalLinear(fc_mid_2),
                    ReLU(inplace=True),
                    fc_end,
                )
            elif case == 'dropout':
                mymodel.linear = nn.Sequential(
                    fc_mid_1,
                    ReLU(inplace=True),
                    Dropout(0.5),
                    fc_mid_2,
                    ReLU(inplace=True),
                    Dropout(0.5),
                    fc_end,
                )
            else:
                assert True
        else:
            assert True

        train_set, test_set = preprocess.cifar10_loaders()
        Learner(model=mymodel, train_loader=train_set, test_loader=test_set,
                seed_str=str(seed), gpu_idx=gpu_idx).run(epochs=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', help='seed', type=int, required=True)
    parser.add_argument('-g', '--gpu', help='gpu_idx', type=int, required=True)
    parser.parse_args()

    main(seed=parser.seed, gpu_idx=parser.gpu)
