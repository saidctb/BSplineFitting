import torch.nn as nn
import torch.nn.functional as f


class SkipConnection(nn.Module):
    def __init__(self, in_ch, out_ch, oto_conv=False, strides=(1, 1)):
        """
        :param in_ch: Number of input channels
        :param out_ch: Number of output channels
        :param oto_conv: Should set to true if change the number of channels
        :param strides: The strides of convolution layers, default is (1,1)
        """
        super(SkipConnection, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, (3, 3), stride=strides, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, (3, 3), padding=1)
        if oto_conv:
            self.conv3 = nn.Conv2d(in_ch, out_ch, (1, 1), stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        out = f.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.conv3:
            x = self.conv3(x)
        out += x
        return out


def residual_block(in_ch, out_ch, num_blocks, first_block=False):
    """
    :param in_ch: Number of input channels
    :param out_ch: Number of output channels
    :param num_blocks: Number of skip connection blocks
    :param first_block: Whether this is the first module in a model
    :return: Sequential of skip connection blocks
    """
    block = []
    for ii in range(num_blocks):
        if ii == 0 and not first_block:
            block.append(SkipConnection(in_ch, out_ch, oto_conv=True, strides=(2, 2)))
        else:
            block.append(SkipConnection(out_ch, out_ch))
    return block


# Definition of the spline model
class SplineModel(nn.Module):
    def __init__(self, num_n):
        super(SplineModel, self).__init__()
        seq1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=3),
                             nn.BatchNorm2d(64), nn.ReLU(),
                             nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        seq2 = nn.Sequential(*residual_block(64, 64, 2, first_block=True))
        seq3 = nn.Sequential(*residual_block(64, 128, 2))
        seq4 = nn.Sequential(*residual_block(128, 256, 2))
        seq5 = nn.Sequential(*residual_block(256, 512, 2))
        self.seq = nn.Sequential(seq1, seq2, seq3, seq4, seq5, nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten(), nn.Linear(512, num_n))

    def forward(self, x):
        out = self.seq(x)
        return out


# Definition of a simpler network
class SimpleSplineModel(nn.Module):
    def __init__(self, num_n):
        super(SimpleSplineModel, self).__init__()
        seq1 = nn.Sequential(nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
                             nn.BatchNorm2d(16), nn.ReLU(),
                             nn.AvgPool2d(kernel_size=(2, 2)))
        seq2 = SkipConnection(16, 16)
        seq3 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
                             nn.BatchNorm2d(16), nn.ReLU(),
                             nn.AvgPool2d(kernel_size=(2, 2)))
        seq4 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
                             nn.BatchNorm2d(16), nn.ReLU(),
                             nn.AvgPool2d(kernel_size=(2, 2)))
        seq5 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
                             nn.BatchNorm2d(16), nn.ReLU(),
                             nn.AvgPool2d(kernel_size=(2, 2)))
        seq6 = nn.Sequential(nn.Flatten(), nn.ReLU(), nn.Linear(4096, 1024),
                             nn.ReLU(), nn.Linear(1024, num_n))
        self.seq = nn.Sequential(seq1, seq2, seq3, seq4, seq5, seq6)

    def forward(self, x):
        out = self.seq(x)
        return out
