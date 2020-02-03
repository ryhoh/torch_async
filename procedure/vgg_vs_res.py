"""
順同期式更新をresnetに適用して、このパラメータ更新手法が他のモデルにおいても有効であることを示す。
先行研究で用いられていたvggについても実験をする。

データセットはImageNetを使い、1000クラス分類をする
"""
import _parent
from os import getcwd
from os.path import join
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

# 引数を受け取る
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--convolution', choices=['resnet', 'vgg_with_maxpool', 'vgg_without_maxpool'],
                    help='convolution type', required=True)
parser.add_argument('-p', '--pooling', choices=['max', 'average'],
                    help='pooling method', required=True)
parser.add_argument('--fc', choices=['none', 'normal', 'semi'],
                    help='full connected type', required=True)

parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batchsize', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learningrate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--gpu', action='store_true', help='flag for Enable GPU')

args = parser.parse_args()
model_type = args.convolution + '_' + args.pooling +\
    '_' + args.fc + '_' + str(args.epochs) + '_' + str(args.batchsize) +\
    '_' + str(args.lr) + '_' + str(args.momentum) + '_' + str(args.seed)
writer = SummaryWriter('runs/' + model_type)


def main():
    global args, writer
    ROOT_DIR = "/Volumes/IMAGENET/ImageNet/"
    TRAIN_CSV = "ILSVRC2012_train.csv"
    VAL_CSV = "ILSVRC2012_val.csv"

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
        model = resnet(args.pooling, args.fc).to(device)
    elif args.convolution == 'vgg_without_maxpool':
        model = vgg(args.convolution, args.pooling, args.fc).to(device)
    elif args.convolution == 'vgg_with_maxpool':
        model = vgg(args.convolution, args.pooling, args.fc).to(device)

    # モデルを記録
    writer.add_graph(model.to('cpu'), torch.randn(1, 3, 224, 224))
    writer.close()
    model = model.to(device)

    # 評価関数
    """ lossはreductionあり/なしを用意する必要がある？

    reduction='mean'(pytorchのデフォルト値)にする
    """
    criterion = CrossEntropyLoss().to(device)

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
    train_loader, val_loader = preprocess.imagenet_dataloaders(
        root_dir=ROOT_DIR,
        train_csv_path=join(ROOT_DIR, TRAIN_CSV),
        val_csv_path=join(ROOT_DIR, VAL_CSV),
        random_seed=args.seed,
        batch_size=args.batchsize
        )

    # 評価
    validate(-1, model, val_loader, criterion, device)

    # 学習
    for epoch in range(args.epochs):
        train(epoch, model, train_loader, optimizer, criterion, device)
        validate(epoch, model, val_loader, criterion, device)
    # 学習結果を保存
    save(model, model_type)


def save(model, name):
    global val_fnames_list, val_outputs_list, val_epoches_list, train_fnames_list, train_outputs_list, train_epoches_list
    """ SAVE MODEL"""
    torch.save(model.state_dict(), join(getcwd(), name+'.model'))
    """ LOAD
    model = TheModelClass(*args, **kwargs)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    """
    """ save predict result """
    pickle.dump({
        "val_fnames_list": val_fnames_list,
        "val_outputs_list": val_outputs_list,
        "val_epoches_list": val_epoches_list,
        "train_fnames_list": train_fnames_list,
        "train_outputs_list": train_outputs_list,
        "train_epoches_list": train_epoches_list}, name+'.pickle')


val_fnames_list = []
val_outputs_list = []
val_epoches_list = []


def validate(epoch, model, val_loader, criterion, device):
    """ 評価用関数 """
    global writer, val_fnames_list, val_outputs_list, val_epoches_list

    model.eval()
    with torch.no_grad():
        fnames_list = []
        outputs_list = []
        loss_sum = 0
        accuracy_sum = 0
        for i, (inputs, labels, fnames) in enumerate(val_loader):
            # デバイス用設定
            inputs = inputs.to(device)
            labels = labels.to(device)
            # モデルへ適用
            outputs = model(inputs)
            # lossを計算
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            # accuracyを計算
            _, argmax = torch.max(outputs, 1)
            accuracy = (labels == argmax.squeeze()).float().mean()
            accuracy_sum += accuracy
            # log
            fnames_list.append(fnames)
            outputs_list.append(outputs)
            # debug
            print('progress: [{0}/{1}]\t'
                  'Loss: {loss:.3f}\t'
                  'Accuracy: {accuracy:.3f}'.format(
                      i, len(val_loader),
                      loss=loss,
                      accuracy=accuracy))
        # output log to tensorboard
        writer.add_scalar('validate loss',
                          loss_sum/len(val_loader),
                          epoch)
        writer.add_scalar('validate Accuracy',
                          accuracy_sum/len(val_loader),
                          epoch)
        # save log
        val_fnames_list.append(fnames_list)
        val_outputs_list.append(outputs_list)
        val_epoches_list.append([[epoch]*len(val_loader)])


train_fnames_list = []
train_outputs_list = []
train_epoches_list = []


def train(epoch, model, train_loader, optimizer, criterion, device):
    """ 学習用関数 """
    global writer, train_fnames_list, train_outputs_list, train_epoches_list

    model.train()
    fnames_list = []
    outputs_list = []
    loss_sum = 0
    accuracy_sum = 0
    for i, (inputs, labels, fnames) in enumerate(train_loader):
        # デバイス用設定
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 勾配の初期化
        optimizer.zero_grad()
        # モデルへ適用
        outputs = model(inputs)
        # lossを計算
        loss = criterion(outputs, labels)
        loss_sum += loss.item()
        # accuracyを計算
        _, argmax = torch.max(outputs, 1)
        accuracy = (labels == argmax.squeeze()).float().mean()
        accuracy_sum += accuracy
        # 逆伝播
        loss.backward()
        # パラメータ更新
        optimizer.step()
        # log
        fnames_list.append(fnames)
        outputs_list.append(outputs)
        # debug
        print('Epoch: [{0}][{1}/{2}]\t'
              'Loss {loss:.4f}\t'
              'Accuracy: {accuracy:.3f}'.format(
               epoch, i, len(train_loader),
               loss=loss.item(),
               accuracy=accuracy))
    # output log to tensorboard
    writer.add_scalar('train loss',
                      loss_sum/len(train_loader),
                      epoch)
    writer.add_scalar('train Accuracy',
                      accuracy_sum/len(train_loader),
                      epoch)
    # save log
    train_fnames_list.append(fnames_list)
    train_outputs_list.append(outputs_list)
    train_epoches_list.append([[epoch]*len(train_loader)])


if __name__ == '__main__':
    main()
