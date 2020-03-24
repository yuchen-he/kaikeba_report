# This file includes the network definition
from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import pdb


class Net_face(nn.Module):
    def __init__(self):
        super(Net_face, self).__init__()

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

        self.block_4_1 = nn.Sequential(OrderedDict([
            ('Conv4_1', nn.Conv2d(24, 40, 3, 1, 1)),
            ('Norm4_1', nn.BatchNorm2d(40, affine=True)),
            ('Relu4_1', nn.PReLU())
        ]))

        self.block_4_2_kp = nn.Sequential(OrderedDict([
            ('Conv4_2_kp', nn.Conv2d(40, 80, 3, 1, 1)),
            ('Norm4_2_kp', nn.BatchNorm2d(80, affine=True)),
            ('Relu4_2_kp', nn.PReLU())
        ]))

        self.ip1_kp = nn.Linear(4 * 4 * 80, 128)
        self.ip2_kp = nn.Linear(128, 128)
        self.landmarks = nn.Linear(128, 42)

        self.block_4_2_cls = nn.Sequential(OrderedDict([
            ('Conv4_2_cls', nn.Conv2d(40, 40, 3, 1, 1)),
            ('Norm4_2_cls', nn.BatchNorm2d(40, affine=True)),
            ('Relu4_2_cls', nn.PReLU())
        ]))

        self.ip1_cls = nn.Linear(4 * 4 * 40, 128)
        self.ip2_cls = nn.Linear(128, 128)
        self.face_cls = nn.Linear(128, 2)
        self.relu = nn.PReLU()

    def forward(self, x):
        # pdb.set_trace()
        # print('input_shape: ', x.shape)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4_1(x)

        kp = self.block_4_2_kp(x)
        kp = kp.view(-1, 4 * 4 * 80)
        kp = self.relu(self.ip1_kp(kp))
        kp = self.relu(self.ip2_kp(kp))
        kp_output = self.landmarks(kp)

        face = self.block_4_2_cls(x)
        face = face.view(-1, 4 * 4 * 40)
        face = self.relu(self.ip1_cls(face))
        face = self.relu(self.ip2_cls(face))
        face_output = self.face_cls(face)
        return kp_output, face_output
