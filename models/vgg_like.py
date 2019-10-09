from torch.nn.modules import Sequential, Conv2d, MaxPool2d, ReLU, ConvTranspose2d, UpsamplingBilinear2d, Tanh
from torch import nn
from torch import Tensor
from torchvision.models.vgg import VGG


def mini_vgg_features():
    model = Sequential(
        # 32 x 32
        Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

        # 16 x 16
        Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

        # 8 x 8
        Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

        # 4 x 4
    )
    return model


# 事前学習用の、対となるデコーダ
def mini_vgg_decoder():
    model = Sequential(
        # 8 x 8
        UpsamplingBilinear2d(scale_factor=2),
        ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        ConvTranspose2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),

        # 16 x 16
        UpsamplingBilinear2d(scale_factor=2),
        ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),

        # 32 x 32
        UpsamplingBilinear2d(scale_factor=2),
        ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        ReLU(),
        ConvTranspose2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        Tanh(),
    )
    return model


def mini_vgg_classifier(hidden_num: int, num_classes: int, dropout: bool) -> nn.Sequential:
    res = nn.Sequential()

    res.add_module('Linear1', nn.Linear(256 * 4 * 4, hidden_num))
    res.add_module('ReLU1', nn.ReLU(True))
    if dropout:
        res.add_module('Dropout1', nn.Dropout())
    res.add_module('Linear2', nn.Linear(hidden_num, hidden_num))
    res.add_module('ReLU2', nn.ReLU(True))
    if dropout:
        res.add_module('Dropout2', nn.Dropout())
    res.add_module('Output', nn.Linear(hidden_num, num_classes))

    return res


class MiniVGG(nn.Module):
    def __init__(self, hidden_num=1024, num_classes=10, init_weights=True, dropout=True):
        super().__init__()
        self.features = mini_vgg_features()
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = mini_vgg_classifier(hidden_num, num_classes, dropout)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class MiniVGG_AE(nn.Module):
    def __init__(self, encoder: nn.Sequential=None, decoder: nn.Sequential=None):
        super().__init__()

        self.encoder = mini_vgg_features() if encoder is None else encoder
        self.decoder = mini_vgg_features() if decoder is None else decoder

    def forward(self, input_tensor) -> Tensor:
        return self.decoder.forward(self.encoder.forward(input_tensor))
