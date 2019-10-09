from torch.nn.modules import Linear
from torchvision.models.vgg import vgg16


def myVGG(out_n: int):
    model = vgg16()
    model.classifier[-1] = Linear(4096, out_n, bias=True)
    return model
