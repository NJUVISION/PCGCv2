# Copyright (c) Nanjing University, Vision Lab.
# Jianqiang Wang (wangjq@smail.nju.edu.cn), Zhan Ma (mazhan@nju.edu.cn); Nanjing University, Vision Lab.
# Last update: 2020.06.06

import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF


class ResNet(nn.Module):
    """
    Basic block: Residual
    """
    
    def __init__(self, channels):
        super(ResNet, self).__init__()
        #path_1
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        
        self.relu = ME.MinkowskiReLU(inplace=True)
        
    def forward(self, x):
        out = self.relu(self.conv0(x))
        out = self.conv1(out)
        out += x

        return out


class InceptionResNet(nn.Module):
    """
    Basic block: Inception Residual.
    """
    
    def __init__(self, channels):
        super(InceptionResNet, self).__init__()
        #path_0
        self.conv0_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels//4,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv0_1 = ME.MinkowskiConvolution(
            in_channels=channels//4,
            out_channels=channels//2,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)

        #path_1
        self.conv1_0 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels//4,
            kernel_size= 1,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_1 = ME.MinkowskiConvolution(
            in_channels=channels//4,
            out_channels=channels//4,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv1_2 = ME.MinkowskiConvolution(
            in_channels=channels//4,
            out_channels=channels//2,
            kernel_size= 1,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)
        
    def forward(self, x):
        out0 = self.conv0_1(self.relu(self.conv0_0(x)))
        out1 = self.conv1_2(self.relu(self.conv1_1(self.relu(self.conv1_0(x)))))
        out = ME.cat(out0, out1) + x

        return out
