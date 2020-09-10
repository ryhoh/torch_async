from models import resnet56
from models import densenet12_40

# citation for ResNet implementation
r"""
@misc{Idelbayev18a,
  author       = "Yerlan Idelbayev",
  title        = "Proper {ResNet} Implementation for {CIFAR10/CIFAR100} in {PyTorch}",
  howpublished = "\url{https://github.com/akamaster/pytorch_resnet_cifar10}",
  note         = "Accessed: 20xx-xx-xx"
}
"""

# citation for DenseNet implementation
r"""
@misc{amos2017densenet,
  title = {{A PyTorch Implementation of DenseNet}},
  author = {Amos, Brandon and Kolter, J. Zico},
  howpublished = {\url{https://github.com/bamos/densenet.pytorch}},
  note = {Accessed: [Insert date here]}
}

Brandon Amos, J. Zico Kolter
A PyTorch Implementation of DenseNet
https://github.com/bamos/densenet.pytorch.
Accessed: [Insert date here]
"""


if __name__ == '__main__':
    # model = resnet56()
    model = densenet12_40()
    print(model)

