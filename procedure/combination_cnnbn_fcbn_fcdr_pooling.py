import _parent
import pickle
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from models.batchnorm_dropout_with_vgg16 import Batchnorm_drouput_with_vgg16 as vgg16
from layers.static import Rotatable
from preprocess import cifar10_dataloader as cifar10
# nan検出
from torch.autograd import detect_anomaly
# ランダムシード
import random
import numpy as np
import torch.backends.cudnn as cudnn
# タイムスタンプ
import datetime
# ファイル操作
from os import getcwd, makedirs
from os.path import join, isdir
# tensorboard
from torch.utils.tensorboard import SummaryWriter
# 引数
import argparse


def rotate_all():
    """ 更新対象を変更"""
    global MODEL

    try:
        for layer in MODEL.fc:
            if isinstance(layer, Rotatable):
                layer.rotate()
    except AttributeError:
        pass


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
            # 更新するニューロンを変更
            rotate_all()

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
    TITLE = "CNNBNとFCBNとFCDR, poolingの組み合わせ実験"
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--convolution",
                        choices=["vgg_with_maxpool", "vgg_without_maxpool"],
                        help="convolution type", required=True)
    parser.add_argument("-p", "--pooling", choices=["max", "average"],
                        help="pooling type", required=True)
    parser.add_argument("--poolingshape", default=7, type=int, metavar="N",
                        help="pooling層の出力サイズ")

    parser.add_argument("--fc", choices=["none", "normal", "semi"],
                        help="fully connected type", required=True)
    parser.add_argument("--deepness", default=2, type=int, metavar="N",
                        help="deepness of fc")

    parser.add_argument("--epochs", default=100, type=int, metavar="N",
                        help="number of Epoch")
    parser.add_argument("--seed", type=int, metavar="N", help="random seed")

    parser.add_argument("--cnn_bn_flag", action="store_true", help="flag for cnn batchnorm")
    parser.add_argument("--fc_bn_flag", action="store_true", help="flag for fc batchnorm")
    parser.add_argument("--fc_do_flag", action="store_true", help="flag for fc dropout")
    parser.add_argument("--nogpu", action="store_true", help="GPU_DISABLED")
    ARGS = parser.parse_args()

    now = datetime.datetime.now()
    MODEL_NAME = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_convolution_among_cnnbn_fcbn_fcdr_pooling".format(
        now.strftime("%Y-%m-%d_%H-%M-%S"),
        ARGS.convolution,
        ARGS.pooling,
        ARGS.poolingshape,
        ARGS.fc,
        ARGS.deepness,
        ARGS.epochs,
        ARGS.seed,
        ARGS.cnn_bn_flag,
        ARGS.fc_bn_flag,
        ARGS.fc_do_flag
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

    """ seed値を固定"""
    # random.seed(seed)
    # np.random.seed(seed)
    # cudnn.deterministic = True
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False

    """ モデル"""
    # モデル
    MODEL = vgg16(num_classes=10, model_type=ARGS.convolution, pooling=ARGS.pooling,
                  poolingshape=ARGS.poolingshape, sync=ARGS.fc,
                  deepness=ARGS.deepness, cnn_bn_flag=ARGS.cnn_bn_flag,
                  fc_bn_flag=ARGS.fc_bn_flag, fc_do_flag=ARGS.fc_do_flag).to(DEVICE)

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
