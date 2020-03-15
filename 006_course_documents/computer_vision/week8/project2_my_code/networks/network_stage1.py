# This file includes the network definition
from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import pdb

class Net_Original(nn.Module):
    def __init__(self):
        super(Net_Original, self).__init__()

        # self.bn = nn.BatchNorm2d(2)
        # self.pool = nn.AvgPool2d(2, 2, ceil_mode=True)

        self.block_1 = nn.Sequential(OrderedDict([
            ('Conv1_1', nn.Conv2d(1, 8, 5, 2, 0)),
            ('Norm1_1', nn.BatchNorm2d(8, affine=True)),
            ('Relu1_1', nn.PReLU()),
            ('AvgPool1', nn.AvgPool2d(2, 2, ceil_mode=True))
        ]))

        self.block_2 = nn.Sequential(OrderedDict([
            ('Conv2_1', nn.Conv2d(8, 16, 3, 1, 0)),
            ('Norm2_1', nn.BatchNorm2d(16, affine=True)),
            ('Relu2_1', nn.PReLU()),
            ('Conv2_2', nn.Conv2d(16, 16, 3, 1, 0)),
            ('Norm2_2', nn.BatchNorm2d(16, affine=True)),
            ('Relu2_2', nn.PReLU()),
            ('AvgPool2', nn.AvgPool2d(2, 2, ceil_mode=True))
        ]))

        self.block_3 = nn.Sequential(OrderedDict([
            ('Conv3_1', nn.Conv2d(16, 24, 3, 1, 0)),
            ('Norm3_1', nn.BatchNorm2d(24, affine=True)),
            ('Relu3_1', nn.PReLU()),
            ('Conv3_2', nn.Conv2d(24, 24, 3, 1, 0)),
            ('Norm3_2', nn.BatchNorm2d(24, affine=True)),
            ('Relu3_2', nn.PReLU()),
            ('AvgPool3', nn.AvgPool2d(2, 2, ceil_mode=True))
        ]))

        self.block_4 = nn.Sequential(OrderedDict([
            ('Conv4_1', nn.Conv2d(24, 40, 3, 1, 1)),
            ('Norm4_1', nn.BatchNorm2d(40, affine=True)),
            ('Relu4_1', nn.PReLU()),
            ('Conv4_2', nn.Conv2d(40, 80, 3, 1, 1)),
            ('Norm4_2', nn.BatchNorm2d(80, affine=True)),
            ('Relu4_2', nn.PReLU())
        ]))

        self.ip1 = nn.Linear(4 * 4 * 80, 128)
        self.ip2 = nn.Linear(128, 128)
        self.ip3 = nn.Linear(128, 42)
        self.relu = nn.PReLU()


    def forward(self, x):
        # pdb.set_trace()
        # print('input_shape: ', x.shape)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        output = x.view(-1, 4 * 4 * 80)

        output = self.relu(self.ip1(output))
        output = self.relu(self.ip2(output))
        output = self.ip3(output)
        return output