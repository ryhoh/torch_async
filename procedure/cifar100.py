"""
順同期式更新をresnetに適用して、このパラメータ更新手法が他のモデルにおいても有効であることを示す。
先行研究で用いられていたvggについても実験をする。

データセットはCifar100を使い、100クラス分類をする
"""
import _parent
import datetime
from os import getcwd, makedirs
from os.path import join, isdir
import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
# DataLoader
import preprocess
# モデル
from models.resnet_for_imagenet import ResNetForImageNet as resnet
from models.vgg_for_imagenet import VGGForImageNet as vgg
# tensorboard
from torch.utils.tensorboard import SummaryWriter
# save list
import pickle
import gc
# debug
from torchsummary import summary

# 引数を受け取る
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--convolution', choices=['resnet', 'vgg_with_maxpool', 'vgg_without_maxpool'],
                    help='convolution type', required=True)
parser.add_argument('-p', '--pooling', choices=['max', 'average'],
                    help='pooling method', required=True)
parser.add_argument('--ps', default=1, type=int,
                    help='pooling shape', required=True)
parser.add_argument('--fc', choices=['none', 'normal', 'semi'],
                    help='full connected type', required=True)
parser.add_argument('--ms', default=4096, type=int,
                    help='middle layer shape', required=True)

parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batchsize', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512)')
parser.add_argument('--lr', '--learningrate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--gpu', action='store_true', help='flag for Enable GPU')
parser.add_argument('--debug', action='store_true', help='flag for Show Model Summary')

args = parser.parse_args()
now = datetime.datetime.now()
if not args.debug:
    model_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_cifar100".format(
        now.strftime("%Y-%m-%d_%H-%M-%S"),
        args.convolution,
        args.pooling,
        str(args.ps),
        args.fc,
        str(args.ms),
        str(args.epochs),
        str(args.batchsize),
        str(args.lr),
        str(args.momentum),
        str(args.seed))
    writer = SummaryWriter('runs/' + model_name)


def main():
    global args, writer, model_name

    # ランダムシード
    if args.seed is not None:
        # PyTorch 以外のRNGを初期化
        random.seed(args.seed)
        np.random.seed(args.seed)
        # cuDNNを使用しない (遅くなる?)
        cudnn.deterministic = True
        # PyTorchのRNGを初期化
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # デバイス設定
    if torch.cuda.is_available() and args.gpu is True:
        device = 'cuda'
    else:
        device = 'cpu'

    # モデル
    if args.convolution == 'resnet':
        model = resnet(
            pooling=args.pooling,
            poolingshape=args.ps,
            sync=args.fc,
            middleshape=args.ms).to(device)
    elif args.convolution == 'vgg_without_maxpool':
        model = vgg(
            model_type=args.convolution,
            pooling=args.pooling,
            poolingshape=args.ps,
            sync=args.fc,
            middleshape=args.ms).to(device)
    elif args.convolution == 'vgg_with_maxpool':
        model = vgg(
            model_type=args.convolution,
            pooling=args.pooling,
            poolingshape=args.ps,
            sync=args.fc,
            middleshape=args.ms).to(device)

    # DEBUG
    if args.debug:
        summary(model, (3, 244, 244))
        exit()

    # モデルを記録
    writer.add_graph(model.to('cpu'), torch.randn(1, 3, 224, 224))
    writer.close()
    model = model.to(device)

    # 評価関数
    """ lossはreductionあり/なしを用意する必要がある？

    reduction='mean'(pytorchのデフォルト値)にする
    """
    criterion_mean = CrossEntropyLoss().to(device)
    criterion_sum = CrossEntropyLoss(reduction='sum').to(device)


    # パラメータ更新手法
    """ SGDのパラメータを設定するか？

    dampening, weight_decay, nesterovは先行研究で設定されてないので未設定
    学習率とモーメンタムは先行研究と同じ値
    """
    optimizer = SGD(model.parameters(),
                    args.lr,
                    momentum=args.momentum)

    # オートチューナーOFF
    torch.backends.cudnn.benchmark = False

    # Dataset
    train_loader, val_loader = preprocess.cifar10_dataloader(
        random_seed=args.seed,
        batch_size=args.batchsize
        )

    # 評価
    validate(-1, model, val_loader, criterion_mean, criterion_sum, device)

    # 学習
    for epoch in range(args.epochs):
        train(epoch, model, train_loader, optimizer, criterion_mean, criterion_sum, device)
        validate(epoch, model, val_loader, criterion_mean, criterion_sum, device)
    # 学習結果を保存
    save(data=model, name=model_name, type="model")


def mkdirs(path):
    """ ディレクトリが無ければ作る """
    if not isdir(path):
        makedirs(path)


def save(data, name, type):
    """ SAVE MODEL

    data: 保存するデータ
    name: ファイル名
    type: データのタイプ
    """
    global model_name
    save_dir = join(getcwd(), "log/" + model_name)

    mkdirs(save_dir)

    if type == "model":
        """ モデルを保存

        Memo: ロードする方法
        model = TheModelClass(*args, **kwargs)
        model.load_state_dict(torch.load(PATH))
        model.eval()
        """
        torch.save(data.state_dict(), join(save_dir, name+'.model'))
    elif type == "progress":
        """ 予測の途中経過

        Memo: ロードする方法
        data = None
        with open(PATH, 'rb') as f:
            data = pickle.load(f)
        """
        with open(join(save_dir, name+'.dump'), 'wb') as f:
            pickle.dump(data, f)


def validate(epoch, model, val_loader, criterion_mean, criterion_sum, device):
    """ 評価用関数 """
    global writer

    model.eval()
    with torch.no_grad():
        outputs_list = []
        loss_sum = 0
        accuracy_sum = 0
        item_counter = 0
        for i, (inputs, labels) in enumerate(val_loader):
            # デバイス用設定
            inputs = inputs.to(device)
            labels = labels.to(device)
            # モデルへ適用
            outputs = model(inputs)
            # lossを計算
            # criterion_meanはbackprop/update用
            loss = criterion_mean(outputs, labels)
            # criterion_sumはログ記録用
            loss_sum += criterion_sum(outputs, labels).item()
            # accuracyを計算
            _, argmax = torch.max(outputs, 1)
            accuracy = (labels == argmax.squeeze()).float().sum().item()
            accuracy_sum += accuracy
            # 画像数
            item_counter += len(outputs)
            # log
            outputs_list.append(outputs.to('cpu'))
            # debug
            print('progress: [{0}/{1}]\t'
                  'Loss: {loss:.3f}\t'
                  'Accuracy: {accuracy:.3f}'.format(
                      i, len(val_loader),
                      loss=loss.item(),
                      accuracy=accuracy/len(outputs)))
        # output log to tensorboard
        writer.add_scalar('validate loss',
                          loss_sum/item_counter,
                          epoch)
        writer.add_scalar('validate Accuracy',
                          accuracy_sum/item_counter,
                          epoch)
        # save log
        d = {
            "outputs": outputs_list
        }
        n = "validate_{}".format(epoch)
        save(data=d, name=n, type="progress")
        del outputs_list
        gc.collect()


def train(epoch, model, train_loader, optimizer, criterion_mean, criterion_sum, device):
    """ 学習用関数 """
    global writer

    model.train()
    outputs_list = []
    loss_sum = 0
    accuracy_sum = 0
    item_counter = 0
    for i, (inputs, labels) in enumerate(train_loader):
        # デバイス用設定
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 勾配の初期化
        optimizer.zero_grad()
        # モデルへ適用
        outputs = model(inputs)
        # lossを計算
        # criterion_meanはbackprop/update用
        loss = criterion_mean(outputs, labels)
        # criterion_sumはログ記録用
        loss_sum += criterion_sum(outputs, labels).item()
        # accuracyを計算
        _, argmax = torch.max(outputs, 1)
        accuracy = (labels == argmax.squeeze()).float().sum().item()
        accuracy_sum += accuracy
        # 画像数
        item_counter += len(outputs)
        # 逆伝播
        loss.backward()
        # パラメータ更新
        optimizer.step()
        # log
        outputs_list.append(outputs.to('cpu'))
        # debug
        print('Epoch: [{0}][{1}/{2}]\t'
              'Loss {loss:.4f}\t'
              'Accuracy: {accuracy:.3f}'.format(
               epoch, i, len(train_loader),
               loss=loss.item(),
               accuracy=accuracy/len(outputs)))
    # output log to tensorboard
    writer.add_scalar('train loss',
                      loss_sum/item_counter,
                      epoch)
    writer.add_scalar('train Accuracy',
                      accuracy_sum/item_counter,
                      epoch)
    # save log
    d = {
        "outputs": outputs_list
    }
    n = "train_{}".format(epoch)
    save(data=d, name=n, type="progress")
    del outputs_list
    gc.collect()


if __name__ == '__main__':
    main()
