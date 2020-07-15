import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
# ランダムシード
import random
import numpy as np
import torch.backends.cudnn as cudnn
# タイムスタンプ
import datetime
# ファイル操作
from os import getcwd, makedirs
from os.path import join, isdir
# nan検出
from torch.autograd import detect_anomaly
# パーセンタイル算出
from typing import Union
from math import floor
# function
from torch.autograd import Function
from torch.nn.functional import linear
# layer
from torch.nn.modules import Linear
# モデル
from torchvision.models import vgg
# データセット
from preprocess import cifar10_dataloader as cifar10
# tensorboard
from torch.utils.tensorboard import SummaryWriter
# save list
import pickle
# 引数
import argparse


def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    k = 1 + floor(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


class Independent_Neuron_Rate_LinearFunction(Function):
    """独立％のLinearFunction
    更新するニューロンをレイヤ内からn％選ぶ(n=√n ,25, 50, 75, 100), またレイヤ間の従属関係はない
    更新するニューロンは重みとバイアスの勾配から決定する
    勾配の絶対値から大きな値を上位n％個選び更新する
    """
    @staticmethod
    def forward(ctx, x, w, b=None, rate=100):
        rate = torch.as_tensor(rate).requires_grad_(False)
        ctx.save_for_backward(x, w, b, rate)
        return linear(x, w, b)

    @staticmethod
    def backward(ctx,grad_out):
        x, w, b, rate = ctx.saved_tensors
        # 勾配計算
        grad_x = grad_out.mm(w)
        grad_w = grad_out.t().mm(x)
        grad_b = grad_out.sum(0)
        # 後で計算に使うマスク(形状がgrad_wで全要素がFalse)
        false_msk = grad_w.abs() < 0

        if 0 <= rate < 100:
            g_w = grad_w.clone()
            g_b = grad_b.clone()
            g_cat = torch.cat((g_w, g_b.unsqueeze(0).T), 1).abs()
            # ニューロンごとに勾配の合計を計算する
            g_sum = torch.sum(g_cat, axis=1)
            # 絶対値の上位n％値を計算
            g_rate = percentile(g_sum, 100-rate)
            # マスクを作る
            b_msk = g_sum <= g_rate
            w_msk = b_msk.unsqueeze(0).T + false_msk
            # 更新するニューロンを選択
            if torch.sum(w_msk) != w_msk.numel():
                grad_w = grad_w.masked_fill(w_msk, 0)
            else:
                grad_w = torch.zeros(grad_w.shape, dtype=torch.float)
            if torch.sum(b_msk) != b_msk.numel():
                grad_b = grad_b.masked_fill(b_msk, 0)
            else:
                grad_b = torch.zeros(grad_b.shape, dtype=torch.float)

        # 例外処理
        if not ctx.needs_input_grad[0]:
            grad_x = None
        if not ctx.needs_input_grad[1]:
            grad_w = None
        if type(b) != torch.Tensor or not ctx.needs_input_grad[2]:
            grad_b = None

        return grad_x, grad_w, grad_b, None


class Independent_Neuron_Rate_Linear(Linear):
    """独立のLinear
    独立更新のレイヤー
    """
    def __init__(self, input_feautures, output_features, bias=True, rate=100):
        super().__init__(input_feautures, output_features, bias)
        self.rate = rate
    def forward(self, input):
        res = Independent_Neuron_Rate_LinearFunction.apply(input, self.weight, self.bias, self.rate)
        return res


def mkdirs(path):
    """ ディレクトリが無ければ作る """
    if not isdir(path):
        makedirs(path)


def save(data, name, task):
    """ SAVE MODEL
    data: 保存するデータ
    name: ファイル名
    task: データのタイプ
    """
    global MODEL_NAME

    save_dir = join(getcwd(), "log/" + MODEL_NAME)
    mkdirs(save_dir)

    if task == "model":
        """ モデルを保存
        Memo: ロードする方法
        model = TheModelClass(*args, **kwargs)
        model.load_state_dict(torch.load(PATH))
        model.eval()
        """
        torch.save(data.state_dict(), join(save_dir, name+'.model'))
    elif task == "progress":
        """ 予測の途中経過
        Memo: ロードする方法
        data = None
        with open(PATH, 'rb') as f:
            data = pickle.load(f)
        """
        with open(join(save_dir, name+'.dump'), 'wb') as f:
            pickle.dump(data, f)


def evaluate(epoch):
    """ 評価"""
    global DEVICE, MODEL, CRITERION_SUM, EVAL_LOADER, WRITER

    MODEL.eval()
    with torch.no_grad():
        loss_sum = 0.0
        accuracy_sum = 0
        item_counter = 0
        answers_list = []
        outputs_list = []

        for i, (inputs, labels) in enumerate(EVAL_LOADER):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            # 推論
            outputs = MODEL(inputs)
            # loss
            loss = CRITERION_MEAN(outputs, labels).item()  # backprop/update用
            loss_sum += CRITERION_SUM(outputs, labels).item()  # 記録用
            # accuracy
            _, argmax = torch.max(outputs, 1)
            accuracy = (labels == argmax.squeeze()).float().sum().item()
            accuracy_sum += accuracy
            # 推論総数
            item_counter += len(outputs)

            # log
            answers_list.append(labels.to('cpu'))
            outputs_list.append(outputs.to('cpu'))
            # debug
            print('progress: [{0}/{1}]\t'
                  'Loss: {loss:.3f}\t'
                  'Accuracy: {accuracy:.3f}'.format(
                      i, len(EVAL_LOADER),
                      loss=loss,
                      accuracy=accuracy/len(outputs)))
        # output log to tensorboard
        WRITER.add_scalar('evaluate loss',
                          loss_sum/item_counter,
                          epoch)
        WRITER.add_scalar('evaluate accuracy',
                          accuracy_sum/item_counter,
                          epoch)
        # save log
        d = {
            "outputs": outputs_list,
            "answers": answers_list
        }
        n = "evaluate_{}".format(epoch)
        save(data=d, name=n, task="progress")


def train(epoch):
    """ 学習"""
    global DEVICE, MODEL, CRITERION_SUM, CRITERION_MEAN, OPTIMIZER,\
        TRAIN_LOADER, WRITER

    MODEL.train()

    loss_sum = 0.0
    accuracy_sum = 0
    item_counter = 0
    answers_list = []
    outputs_list = []

    with detect_anomaly():
        for i, (inputs, labels) in enumerate(TRAIN_LOADER):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            # 勾配を初期化
            OPTIMIZER.zero_grad()
            # 推論
            outputs = MODEL(inputs)
            # loss
            loss = CRITERION_MEAN(outputs, labels)  # backprop/update用
            loss_sum += CRITERION_SUM(outputs, labels).item()  # 記録用
            # accuracy
            _, argmax = torch.max(outputs, 1)
            accuracy = (labels == argmax.squeeze()).float().sum().item()
            accuracy_sum += accuracy
            # 推論総数
            item_counter += len(outputs)

            # 逆伝播
            loss.backward()
            # パラメータ更新
            OPTIMIZER.step()

            # log
            answers_list.append(labels.to('cpu'))
            outputs_list.append(outputs.to('cpu'))
            # debug
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.3f}\t'
                  'Accuracy: {accuracy:.3f}'.format(
                   epoch, i, len(TRAIN_LOADER),
                   loss=loss.item(),
                   accuracy=accuracy/len(outputs)))
        # output log to tensorboard
        WRITER.add_scalar('train loss',
                          loss_sum/item_counter,
                          epoch)
        WRITER.add_scalar('train accuracy',
                          accuracy_sum/item_counter,
                          epoch)
        # save log
        d = {
            "outputs": outputs_list,
            "answers": answers_list
        }
        n = "train_{}".format(epoch)
        save(data=d, name=n, task="progress")


if __name__ == "__main__":
    # 引数を受け取る
    TITLE = "勾配の大きさを基準に更新するパラメータ選ぶ独立準同期式更新"
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--epochs", default=50, type=int, metavar="N",
                        help="number of Epoch")
    parser.add_argument("--rate", type=float, metavar="N", help="更新率")
    parser.add_argument("--seed", type=int, metavar="N", help="random seed")
    parser.add_argument("--nogpu", action="store_true", help="GPU_DISABLED")
    ARGS = parser.parse_args()

    now = datetime.datetime.now()
    MODEL_NAME = "{}_{}_{}_independent_neuron_rate".format(
        now.strftime("%Y-%m-%d_%H-%M-%S"),
        ARGS.epochs,
        ARGS.rate,
        ARGS.seed
    )
    WRITER = SummaryWriter("runs/" + MODEL_NAME)

    # デバイス設定
    DEVICE = "cpu"
    # デバイス設定
    if torch.cuda.is_available() and not ARGS.nogpu:
        DEVICE = 'cuda'

    """ 学習パラメータを定義"""
    seed = ARGS.seed
    epochs = ARGS.epochs
    momentum = 0.9
    lr = 0.001
    batch_size = 32
    per = ARGS.rate

    """ seed値を固定"""
    random.seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

    """ モデル"""
    MODEL = vgg.vgg16()
    # 準同期式に変更
    poolingshape = 7
    in_shape = 512 * poolingshape * poolingshape
    middleshape = 4096
    outshape = 10
    MODEL.classifier[0] = Independent_Neuron_Rate_Linear(in_shape, middleshape, True, per)
    MODEL.classifier[3] = Independent_Neuron_Rate_Linear(
        middleshape, middleshape, True, per)
    MODEL.classifier[-1] = Linear(middleshape, outshape)
    MODEL = MODEL.to(DEVICE)

    # 評価関数
    CRITERION_MEAN = CrossEntropyLoss().to(DEVICE)
    CRITERION_SUM = CrossEntropyLoss(reduction='sum').to(DEVICE)
    # 最適化手法
    OPTIMIZER = SGD(MODEL.parameters(), lr, momentum=momentum)

    """ datasetを定義"""
    TRAIN_LOADER, EVAL_LOADER = cifar10(
        random_seed=seed, batch_size=batch_size)

    """ 学習フロー"""
    # 評価
    evaluate(-1)
    # 学習
    for epoch in range(epochs):
        train(epoch)
        evaluate(epoch)
    # 学習結果を保存
    save(data=MODEL, name=MODEL_NAME, task="model")
    print("done")
