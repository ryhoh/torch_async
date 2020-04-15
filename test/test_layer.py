from copy import deepcopy
import unittest

import torch
from torch import tensor
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn.modules import Linear, Conv2d, MSELoss
from torch.nn.functional import relu
from torchvision import models

from layers import SemisyncLinear, MyConv2d, RandomSemiSyncConv2d


class TestSemiSyncNet(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.vgg16 = models.vgg16().to('cuda')
        self.vgg16.classifier[-1] = Linear(4096, 10, bias=True).to('cuda')
        self.vgg16_semisync = deepcopy(self.vgg16)

    # 同期式のレイヤーから、パラメータをそのままにして準同期式レイヤを作る
    def testParam(self):
        self.vgg16_semisync.classifier[0] = SemisyncLinear(
            self.vgg16_semisync.classifier[0],
        ).to('cuda')
        self.vgg16_semisync.classifier[3] = SemisyncLinear(
            self.vgg16_semisync.classifier[3],
        ).to('cuda')

        # 別オブジェクトだが同じパラメータを持つことを確認
        self.assertTrue(self.vgg16 is not self.vgg16_semisync)

        for layer1, layer2 in zip(self.vgg16.classifier, self.vgg16_semisync.classifier):
            self.assertTrue(isinstance(layer1, Linear) == isinstance(layer2, Linear))

            if isinstance(layer1, Linear) and isinstance(layer2, Linear):
                self.assertTrue((layer1.weight == layer2.weight).byte().all())
                self.assertTrue((layer1.bias == layer2.bias).byte().all())


class TestSemiSync(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        base_layer = Linear(2, 4)
        base_layer.weight = Parameter(tensor([
            [1.7, 0.4, 1, 2.2],
            [1.8, -1, 0.9, -0.2]
        ], requires_grad=True).t())
        base_layer.bias = Parameter(tensor([0.0, 0.0, 0.0, 0.0], requires_grad=True))

        self.layer = SemisyncLinear(base_layer, group_list=None).to('cuda')

    def testGroup(self):
        self.assertEqual([0, 2, 4], self.layer.group_delim)

    def testForward(self):
        X = tensor([  # データ数3と仮定, shape=(3, 2)
            [1.2, 1.4],
            [-0.8, 0.2],
            [1.1, -1.2]
        ]).to('cuda')
        expected = tensor([
            [4.56, -0.92, 2.46, 2.36],
            [-1, -0.52, -0.62, -1.8],
            [-0.29, 1.64, 0.02, 2.66]
        ]).to('cuda')
        actual = self.layer.forward(X)
        self.assertTrue(torch.all(torch.abs(expected - actual) < 10e-6))

    def testBackward(self) -> None:
        X = Variable(tensor([  # データ数3と仮定, shape=(3, 2)
            [1.2, 1.4],
            [-0.8, 0.2],
            [1.1, -1.2]
        ], requires_grad=True).to('cuda'))
        X.requires_grad_()

        expected_dx = tensor([
            [5.3, 1.5],
            [5.3, 1.5],
            [5.3, 1.5]
        ]).to('cuda')

        # -- neuron 0,1 get gradient --
        expected_dW = tensor([
            [1.5, 1.5, 0, 0],
            [0.4, 0.4, 0, 0]
        ]).t().to('cuda')
        expected_db = tensor([3.0, 3.0, 0, 0]).to('cuda')

        dout = self.layer.forward(X)
        dout.backward(torch.ones((3, 4)).to('cuda'))

        self.assertTrue(
            torch.all(torch.abs(expected_dx - X.grad) < 10e-6).item())
        self.assertTrue(
            torch.all(torch.abs(expected_dW - self.layer.weight.grad) < 10e-6).item())
        self.assertTrue(
            torch.all(torch.abs(expected_db - self.layer.bias.grad) < 10e-6).item())

        self.layer.rotate()

        # clear gradients
        self.layer.weight.grad = None
        self.layer.bias.grad = None
        X.grad = None

        # -- neuron 2,3 get gradient --
        expected_dW = tensor([
            [0, 0, 1.5, 1.5],
            [0, 0, 0.4, 0.4]
        ]).t().to('cuda')
        expected_db = tensor([0, 0, 3.0, 3.0]).to('cuda')

        dout = self.layer.forward(X)
        dout.backward(torch.ones((3, 4)).to('cuda'))

        self.assertTrue(
            torch.all(torch.abs(expected_dx - X.grad) < 10e-6).item())
        self.assertTrue(
            torch.all(torch.abs(expected_dW - self.layer.weight.grad) < 10e-6).item())
        self.assertTrue(
            torch.all(torch.abs(expected_db - self.layer.bias.grad) < 10e-6).item())

        self.layer.rotate()

        # clear gradients
        self.layer.weight.grad = None
        self.layer.bias.grad = None
        X.grad = None

        # -- neuron 0,1 get gradient again --
        expected_dW = tensor([
            [1.5, 1.5, 0, 0],
            [0.4, 0.4, 0, 0]
        ]).t().to('cuda')
        expected_db = tensor([3.0, 3.0, 0, 0]).to('cuda')

        dout = self.layer.forward(X)
        dout.backward(torch.ones((3, 4)).to('cuda'))
        self.assertTrue(
            torch.all(torch.abs(expected_dx - X.grad) < 10e-6).item())
        self.assertTrue(
            torch.all(torch.abs(expected_dW - self.layer.weight.grad) < 10e-6).item())
        self.assertTrue(
            torch.all(torch.abs(expected_db - self.layer.bias.grad) < 10e-6).item())

        self.layer.rotate()

    def testRotate(self):
        X = tensor([  # データ数3と仮定, shape=(3, 2)
                    [1.2, 1.4],
                    [-0.8, 0.2],
                    [1.1, -1.2]
        ], requires_grad=True).to('cuda')

        self.layer.forward(X)
        left = self.layer.learn_l
        right = self.layer.learn_r
        self.assertEqual(0, left)
        self.assertEqual(2, right)
        self.layer.rotate()

        self.layer.forward(X)
        left = self.layer.learn_l
        right = self.layer.learn_r
        self.assertEqual(2, left)
        self.assertEqual(4, right)
        self.layer.rotate()

        self.layer.forward(X)
        left = self.layer.learn_l
        right = self.layer.learn_r
        self.assertEqual(0, left)
        self.assertEqual(2, right)
        self.layer.rotate()


class TestSemiSyncAtUnbalance(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.layer = SemisyncLinear(Linear(4000, 4000), group_list=None).to('cuda')

    def testGroup(self):
        expected = [63 * i for i in range(64)] + [4000]
        self.assertEqual(expected, self.layer.group_delim)

    def testRotate(self):
        X = torch.ones((1, 4000)).to('cuda')

        for i in range(63):
            self.layer.forward(X)
            left = self.layer.learn_l
            right = self.layer.learn_r
            self.assertEqual(63 * i, left)
            self.assertEqual(63 * (i + 1), right)
            self.layer.rotate()

        self.layer.forward(X)
        left = self.layer.learn_l
        right = self.layer.learn_r
        self.assertEqual(3969, left)
        self.assertEqual(4000, right)
        self.layer.rotate()

        self.layer.forward(X)
        left = self.layer.learn_l
        right = self.layer.learn_r
        self.assertEqual(0, left)
        self.assertEqual(63, right)
        self.layer.rotate()


class TestMyConv2d(unittest.TestCase):
    def setUp(self) -> None:
        self.orig_conv1 = Conv2d( 3,  32, kernel_size=9, stride=1)
        self.orig_conv2 = Conv2d(32,  64, kernel_size=3, stride=2)
        self.orig_conv3 = Conv2d(64, 128, kernel_size=3, stride=2)

        self.conv1 = MyConv2d(self.orig_conv1)
        self.conv2 = MyConv2d(self.orig_conv2)
        self.conv3 = MyConv2d(self.orig_conv3)

    def testParam(self):
        # 別オブジェクトだが同じパラメータを持つことを確認
        self.assertTrue(self.orig_conv1 is not self.conv1)
        self.assertTrue(self.orig_conv2 is not self.conv2)
        self.assertTrue(self.orig_conv3 is not self.conv3)

        self.assertTrue(self.orig_conv1.in_channels  == self.conv1.in_channels)
        self.assertTrue(self.orig_conv1.out_channels == self.conv1.out_channels)
        self.assertTrue(self.orig_conv1.kernel_size  == self.conv1.kernel_size)
        self.assertTrue(self.orig_conv1.stride       == self.conv1.stride)
        self.assertTrue(self.orig_conv1.padding      == self.conv1.padding)
        self.assertTrue(self.orig_conv1.dilation     == self.conv1.dilation)
        self.assertTrue(self.orig_conv1.groups       == self.conv1.groups)
        self.assertTrue(self.orig_conv1.padding_mode == self.conv1.padding_mode)

        self.assertTrue((self.orig_conv1.weight == self.conv1.weight).byte().all())
        self.assertTrue((self.orig_conv1.bias   == self.conv1.bias  ).byte().all())
        self.assertTrue((self.orig_conv2.weight == self.conv2.weight).byte().all())
        self.assertTrue((self.orig_conv2.bias   == self.conv2.bias  ).byte().all())
        self.assertTrue((self.orig_conv3.weight == self.conv3.weight).byte().all())
        self.assertTrue((self.orig_conv3.bias   == self.conv3.bias  ).byte().all())

    def testForward(self):
        torch.manual_seed(0)
        X = torch.randn(50, 3, 128, 128)  # (128 x 128) カラー画像 50枚

        expected_y = relu(self.orig_conv1.forward(X))
        expected_y = relu(self.orig_conv2.forward(expected_y))
        expected_y = relu(self.orig_conv3.forward(expected_y))

        actual_y = relu(self.conv1.forward(X))
        actual_y = relu(self.conv2.forward(actual_y))
        actual_y = relu(self.conv3.forward(actual_y))

        self.assertTrue((expected_y == actual_y).byte().all())


class TestRandomSemiSyncConv2d(unittest.TestCase):
    def setUp(self) -> None:
        self.orig_conv1 = Conv2d( 3,  32, kernel_size=9, stride=1)
        self.orig_conv2 = Conv2d(32,  64, kernel_size=3, stride=2)
        self.orig_conv3 = Conv2d(64, 128, kernel_size=3, stride=2)

        self.conv1 = RandomSemiSyncConv2d(self.orig_conv1, 1.0)
        self.conv2 = RandomSemiSyncConv2d(self.orig_conv2, 1.0)
        self.conv3 = RandomSemiSyncConv2d(self.orig_conv3, 1.0)

    def testForward(self):
        torch.manual_seed(0)
        X = torch.randn(50, 3, 128, 128)  # (128 x 128) カラー画像 50枚

        expected_y = relu(self.orig_conv1.forward(X))
        expected_y = relu(self.orig_conv2.forward(expected_y))
        expected_y = relu(self.orig_conv3.forward(expected_y))

        actual_y = relu(self.conv1.forward(X))
        actual_y = relu(self.conv2.forward(actual_y))
        actual_y = relu(self.conv3.forward(actual_y))

        self.assertTrue((expected_y == actual_y).byte().all())


if __name__ == '__main__':
    unittest.main()
