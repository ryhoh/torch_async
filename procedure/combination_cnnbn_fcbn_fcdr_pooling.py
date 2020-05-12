import _parent
import pickle
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from models.batchnorm_dropout_with_vgg16 import Batchnorm_drouput_with_vgg16 as vgg16
from layers.static import Rotatable
import preprocess
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
# メモリ解放
import gc
# 引数
import argparse

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
if torch.cuda.is_available() and ARGS.nogpu is False:
    DEVICE = "cuda"


# ランダムシードを固定
if ARGS.seed is not None:
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    cudnn.deterministic = True
    torch.manual_seed(ARGS.seed)
    torch.cuda.manual_seed(ARGS.seed)
    torch.backends.cudnn.benchmark = False


def mkdirs(path):
    """ ディレクトリを作る"""
    if not isdir(path):
        makedirs(path)


def save(data, name, type):
    """ 保存する

    data: 保存対象のデータ
    name: ファイル名
    task: データタイプ
    """
    global MODEL_NAME

    save_dir = join(getcwd(), "log/" + MODEL_NAME)

    mkdirs(save_dir)

    if type == "model":
        """ モデルのパラメータを保存

        Memo: ロードする方法
        model = TheModelClass(*args, **kwargs)
        model.load_state_dict(torch.load(PATH))
        model.eval()
        """
        torch.save(data.state_dict(), join(save_dir, name+'.model'))
    elif type == "prediction":
        """ 予測値

        Model: ロードする方法
        data = none
        with open(PATH, "rb") as f:
            data = pickle.load(f)
        """
        with open(join(save_dir, name+'.dump'), 'wb') as f:
            pickle.dump(data, f)
    else:
        raise ValueError("taskの値が不正です")


def rotate_all(model):
    try:
        for layer in model.fc:
            if isinstance(layer, Rotatable):
                layer.rotate()
    except AttributeError:
        pass


def evaluate(epoch, model, dataloader,  criterion_sum, criterion_mean):
    """ モデルの評価"""
    global WRITER, DEVICE

    model.eval()
    with torch.no_grad():
        outputs_list = []
        answers_list = []

        loss_sum = 0.0
        accuracy_sum = 0
        item_counter = 0

        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            # 推論
            outputs = model(inputs)
            # loss
            loss = criterion_mean(outputs, labels)  # backprop/update用
            loss_sum += criterion_sum(outputs, labels).item()  # 記録用
            # accuracy
            _, argmax = torch.max(outputs, 1)
            accuracy = (labels == argmax.squeeze()).float().sum().item()
            accuracy_sum += accuracy
            # 推論総数
            item_counter += len(outputs)
            # 推論結果と対応する答えを記録
            outputs_list.append(outputs.to('cpu'))
            answers_list.append(outputs.to('cpu'))

            # debug
            print("progress: [{0}/{1}]\t"
                  "Loss: {loss:.3f}\t"
                  "Acc: {accuracy:.3f}".format(
                        i, len(dataloader),
                        loss=loss.item(),
                        accuracy=accuracy/len(outputs)))
    # 記録
    WRITER.add_scalar("evaluate loss", loss_sum/item_counter, epoch)
    WRITER.add_scalar("evaluate acc", accuracy_sum/item_counter, epoch)
    d = {
        "outputs": outputs_list,
        "answers": answers_list
    }
    n = "evaluate_{}".format(epoch)
    save(data=d, name=n, type="prediction")
    # メモリ解放
    del outputs_list
    del answers_list
    gc.collect()


def train(epoch, model, dataloader, optimizer, criterion_sum, criterion_mean):
    """ モデルの学習"""
    global WRITER, DEVICE

    model.train()
    outputs_list = []
    answers_list = []

    loss_sum = 0.0
    accuracy_sum = 0
    item_counter = 0

    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        # 勾配を初期化
        optimizer.zero_grad()
        # 推論
        outputs = model(inputs)
        # loss
        loss = criterion_mean(outputs, labels)  # backprop/update用
        loss_sum += criterion_sum(outputs, labels)  # 記録用
        # accuracy
        _, argmax = torch.max(outputs, 1)
        accuracy = (labels == argmax.squeeze()).float().sum().item()
        accuracy_sum += accuracy
        # 推論総数
        item_counter += len(outputs)
        # 推論結果と対応する答えを記録
        outputs_list.append(outputs.to('cpu'))
        answers_list.append(outputs.to('cpu'))

        # 逆伝播
        loss.backward()
        # パラメータ更新
        optimizer.step()

        # ニューロン切り替え
        rotate_all(model)

        # debug
        print("Epoch: [{0}][{1}/{2}]\t"
              "Loss: {loss:.4f}\t"
              "Acc: {accuracy:.3f}".format(
                    epoch, i, len(dataloader),
                    loss=loss.item(),
                    accuracy=accuracy/len(outputs)))

    # 記録
    WRITER.add_scalar("train loss", loss_sum/item_counter, epoch)
    WRITER.add_scalar("train acc", accuracy_sum/item_counter, epoch)
    d = {
        "outputs": outputs_list,
        "answers": answers_list
    }
    n = "train_{}".format(epoch)
    save(data=d, name=n, type="prediction")
    # メモリ解放
    del outputs_list
    del answers_list
    gc.collect()


def main():
    """NOTE: 準同期式更新が使えるようにfcレイヤの名前はfcにするclassifierとかはだめ"""
    """poolingshapeを適切に指定
    if ARGS.convolution == "vgg16":
        poolingshape = 7
    elif ARGS.convolution == "resnet18":
        poolingshape = 1
    """
    """cifar10はnumclass10"""
    """inceptionv3はresolution=299"""
    global ARGS, DEVICE, MODEL_NAME

    # モデル
    model = vgg16(num_classes=10, model_type=ARGS.convolution, pooling=ARGS.pooling,
                  poolingshape=ARGS.poolingshape, sync=ARGS.fc,
                  deepness=ARGS.deepness, cnn_bn_flag=ARGS.cnn_bn_flag,
                  fc_bn_flag=ARGS.fc_bn_flag, fc_do_flag=ARGS.fc_do_flag).to(DEVICE)

    # 評価関数
    criterion_sum = CrossEntropyLoss().to(DEVICE)
    criterion_mean = CrossEntropyLoss(reduction='sum').to(DEVICE)
    # 最適化手法
    optimizer = SGD(model.parameters(), 0.001, momentum=0.9)

    # dataloader
    train_dataloader, eval_dataloader = preprocess.cifar10_dataloader(
        random_seed=ARGS.seed, batch_size=32)

    # 評価
    evaluate(-1, model, eval_dataloader,
             criterion_sum=criterion_sum, criterion_mean=criterion_mean)
    # 学習
    for epoch in range(ARGS.epochs):
        train(epoch, model, train_dataloader, optimizer,
              criterion_sum=criterion_sum, criterion_mean=criterion_mean)
        evaluate(epoch, model, eval_dataloader,
                 criterion_sum=criterion_sum, criterion_mean=criterion_mean)
    # 学習結果を記録
    save(data=model, name=MODEL_NAME, type="model")


if __name__ == "__main__":
    main()
