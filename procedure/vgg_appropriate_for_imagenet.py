""" vggはimagenetでも使えるか？"""
import _parent
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg
from torchvision import models
from layers.static import Rotatable, OptimizedSemiSyncLinear
# データセット
from dataloaders.imagenet import imagenet_train_eval_dataloaders as imagenet
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
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--convolution', choices=['resnet18', 'vgg16'],
                    help='convolution type', required=True)
parser.add_argument('--fc', choices=['none', 'normal', 'semi'],
                    help='full connected type', required=True)
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
args = parser.parse_args()

now = datetime.datetime.now()
MODEL_NAME = "{}_{}_{}_{}_hori_modif_Imagenet".format(
    now.strftime("%Y-%m-%d_%H-%M-%S"),
    args.convolution,
    args.fc,
    args.epochs
)
writer = SummaryWriter('runs/' + MODEL_NAME)

GPU_ENABLED = True


def conduct(model: nn.Module, train_loader, test_loader, a_loader) -> dict:
    def rotate_all():
        global args, model

        try:
            if args.convolution == "resnet18":
                for layer in model.fc:
                    if isinstance(layer, Rotatable):
                        layer.rotate()
            elif args.convolution == "vgg16":
                for layer in model.classifier:
                    if isinstance(layer, Rotatable):
                        layer.rotate()
                pass
            else:
                raise ValueError("引数convolutionの値が不正です")
        except AttributeError:
            pass

    loss_layer = nn.CrossEntropyLoss(reduction='none')
    loss_layer_reduce = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # validation of pre_training
    validate(model, test_loader, -1)
    validate_a(model, a_loader, -1)

    # training
    for epoch in range(args.epochs):
        model.train()
        outputs_list = []
        fnames_list = []
        total_loss = 0.0
        total_correct = 0
        item_counter = 0

        for i, mini_batch in enumerate(train_loader):
            input_data, label_data, fnames = mini_batch

            if GPU_ENABLED:
                in_tensor = input_data.to('cuda')
                label_tensor = label_data.to('cuda')
            else:
                in_tensor = input_data
                label_tensor = label_data

            optimizer.zero_grad()  # Optimizer を0で初期化

            # forward - backward - optimize
            outputs = model(in_tensor)
            loss_vector = loss_layer(outputs, label_tensor)  # for evaluation
            reduced_loss = loss_layer_reduce(
                outputs, label_tensor)  # for backward
            _, predicted = torch.max(outputs.data, 1)

            reduced_loss.backward()
            optimizer.step()

            rotate_all()

            total_loss += loss_vector.data.sum().item()
            total_correct += (predicted.to('cpu') == label_data).sum().item()
            item_counter += len(outputs)
            outputs_list.append(outputs.to('cpu'))
            fnames_list.append(fnames)
            # debug
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f}\t'
                  'Accuracy: {accuracy:.3f}'.format(
                   epoch, i, len(train_loader),
                   loss=loss_vector.data.sum().item(),
                   accuracy=(predicted.to('cpu') == label_data).sum().item()/len(outputs)))

        """ 記録"""
        writer.add_scalar('train loss',
                          total_loss/item_counter,
                          epoch)
        writer.add_scalar('train Accuracy',
                          total_correct/item_counter,
                          epoch)
        d = {
            "outputs": outputs_list,
            "file_names": fnames_list
        }
        n = "train_{}".format(epoch)
        save(data=d, name=n, type="progress")
        """ メモリ解放"""
        del outputs_list
        del fnames_list
        gc.collect()

        """ 評価"""
        validate(model, test_loader, epoch)
        validate_a(model, a_loader, epoch)

    print('Finished Training')


def validate_a(model: nn.Module, test_loader, epoch: int):
    model.eval()
    with torch.no_grad():
        loss_layer = nn.CrossEntropyLoss(reduction='none')

        outputs_list = []
        fnames_list = []
        total_correct = 0
        total_loss = 0.0
        item_counter = 0

        for i, mini_batch in enumerate(test_loader):
            input_data, label_data, fnames = mini_batch
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
            item_counter += len(outputs)
            outputs_list.append(outputs.to('cpu'))
            fnames_list.append(fnames)
            # debug
            print('progress: [{0}/{1}]\t'
                  'Loss: {loss:.3f}\t'
                  'Accuracy: {accuracy:.3f}'.format(
                      i, len(test_loader),
                      loss=loss_vector.sum().item(),
                      accuracy=(predicted.to('cpu') == label_data).sum().item()/len(outputs)))

        """ 記録"""
        writer.add_scalar('a loss',
                          total_loss/item_counter,
                          epoch)
        writer.add_scalar('a Accuracy',
                          total_correct/item_counter,
                          epoch)
        d = {
            "outputs": outputs_list,
            "file_names": fnames_list
        }
        n = "a_{}".format(epoch)
        save(data=d, name=n, type="progress")
        """ メモリ解放"""
        del outputs_list
        del fnames_list
        gc.collect()


def validate(model: nn.Module, test_loader, epoch: int):
    model.eval()
    with torch.no_grad():
        loss_layer = nn.CrossEntropyLoss(reduction='none')

        outputs_list = []
        fnames_list = []
        total_correct = 0
        total_loss = 0.0
        item_counter = 0

        for i, mini_batch in enumerate(test_loader):
            input_data, label_data, fnames = mini_batch
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
            item_counter += len(outputs)
            outputs_list.append(outputs.to('cpu'))
            fnames_list.append(fnames)
            # debug
            print('progress: [{0}/{1}]\t'
                  'Loss: {loss:.3f}\t'
                  'Accuracy: {accuracy:.3f}'.format(
                      i, len(test_loader),
                      loss=loss_vector.sum().item(),
                      accuracy=(predicted.to('cpu') == label_data).sum().item()/len(outputs)))

        """ 記録"""
        writer.add_scalar('validate loss',
                          total_loss/item_counter,
                          epoch)
        writer.add_scalar('validate Accuracy',
                          total_correct/item_counter,
                          epoch)
        d = {
            "outputs": outputs_list,
            "file_names": fnames_list
        }
        n = "validate_{}".format(epoch)
        save(data=d, name=n, type="progress")
        """ メモリ解放"""
        del outputs_list
        del fnames_list
        gc.collect()


def mkdirs(path):
    """ ディレクトリが無ければ作る """
    if not isdir(path):
        makedirs(path)


def save(data, name, type):
    """ 保存

    data: 保存するデータ
    name: ファイル名
    type: データのタイプ
    """
    global MODEL_NAME

    save_dir = join(getcwd(), "log/" + MODEL_NAME)
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


if __name__ == '__main__':
    torch.manual_seed(0)
    if args.convolution == "vgg16":
        poolingshape = 7
        in_shape = 512 * poolingshape * poolingshape
        middleshape = 4096
        num_classes = 10
    elif args.convolution == "resnet18":
        poolingshape = 1
        in_shape = 512 * poolingshape * poolingshape
        middleshape = 4096
        num_classes = 200
    else:
        raise ValueError("引数convolutionの値が不正です")

    # モデルを定義
    if args.convolution == "vgg16":
        model = vgg.vgg16(pretrained=False)
        if args.fc == "normal":
            model.classifier = nn.Sequential(
                nn.Linear(in_shape, middleshape),
                nn.ReLU(middleshape),
                nn.Dropout(p=0.5),
                nn.Linear(middleshape, middleshape),
                nn.ReLU(middleshape),
                nn.Dropout(p=0.5),
                nn.Linear(middleshape, num_classes)
            )
        elif args.fc == "semi":
            model.classifier = nn.Sequential(
                OptimizedSemiSyncLinear(nn.Linear(in_shape, middleshape)),
                nn.ReLU(middleshape),
                nn.Dropout(p=0.5),
                OptimizedSemiSyncLinear(nn.Linear(middleshape, middleshape)),
                nn.ReLU(middleshape),
                nn.Dropout(p=0.5),
                nn.Linear(middleshape, num_classes)
            )
        elif args.fc == "none":
            model.classifier = nn.Sequential(
                nn.Linear(in_shape, num_classes),
            )
        else:
            raise ValueError("引数fcの値が不正です")
    elif args.convolution == "resnet18":
        model = models.resnet18(pretrained=False)
        if args.fc == "normal":
            model.fc = nn.Sequential(
                nn.Linear(in_shape, middleshape),
                nn.ReLU(middleshape),
                nn.Dropout(p=0.5),
                nn.Linear(middleshape, middleshape),
                nn.ReLU(middleshape),
                nn.Dropout(p=0.5),
                nn.Linear(middleshape, num_classes)
            )
        elif args.fc == "semi":
            model.fc = nn.Sequential(
                OptimizedSemiSyncLinear(nn.Linear(in_shape, middleshape)),
                nn.ReLU(middleshape),
                nn.Dropout(p=0.5),
                OptimizedSemiSyncLinear(nn.Linear(middleshape, middleshape)),
                nn.ReLU(middleshape),
                nn.Dropout(p=0.5),
                nn.Linear(middleshape, num_classes)
            )
        elif args.fc == "none":
            model.fc = nn.Sequential(
                nn.Linear(in_shape, num_classes),
            )
        else:
            raise ValueError("引数fcの値が不正です")
    else:
        raise ValueError("引数convolutionの値が不正です")

    if GPU_ENABLED:
        model.to('cuda')
    else:
        model.to('cpu')

    ROOT_DIR = "/ImageNet/"
    TRAIN_CSV = "csvs/imagenet_train_200.csv"
    VAL_CSV = "csvs/imagenet_val_200.csv"
    A_CSV = "csvs/imagenet_a_200.csv"

    conduct(model, *(imagenet(
        root_dir=ROOT_DIR,
        train_csv_path=join(ROOT_DIR, TRAIN_CSV),
        val_csv_path=join(ROOT_DIR, VAL_CSV),
        a_csv_path=join(ROOT_DIR, A_CSV),
        random_seed=0,
        batch_size=32,
        resolution=224
        )))

    """ 学習後のモデルをdumpする"""
    save(data=model, name=MODEL_NAME, type="model")
