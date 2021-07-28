import torch
import MinkowskiEngine as ME

from data_utils import isin, istopk


class InceptionResNet(torch.nn.Module):
    """Inception Residual Network
    """
    
    def __init__(self, channels):
        super().__init__()
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

def make_layer(block, block_layers, channels):
    """make stacked InceptionResNet layers.
    """
    layers = []
    for i in range(block_layers):
        layers.append(block(channels=channels))
        
    return torch.nn.Sequential(*layers)

class Encoder(torch.nn.Module):
    def __init__(self, channels=[1,16,32,64,32,8]):
        super().__init__()
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down0 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.block0 = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels[2])

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.block1 = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels[3])

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down2 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[4],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.block2 = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels[4])

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=channels[4],
            out_channels=channels[5],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        out0 = self.relu(self.down0(self.relu(self.conv0(x))))
        out0 = self.block0(out0)
        out1 = self.relu(self.down1(self.relu(self.conv1(out0))))
        out1 = self.block1(out1)
        out2 = self.relu(self.down2(self.relu(self.conv2(out1))))
        out2 = self.block2(out2)
        out2 = self.conv3(out2)

        return [out2, out1, out0]


class Decoder(torch.nn.Module):
    """the decoding network with upsampling.
    """
    def __init__(self, channels=[8,64,32,16]):
        super().__init__()
        self.up0 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=channels[1],
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        self.block0 = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels[1])

        self.conv0_cls = ME.MinkowskiConvolution(
            in_channels=channels[1],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.up1 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=channels[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.block1 = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels[2])

        self.conv1_cls = ME.MinkowskiConvolution(
            in_channels=channels[2],
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.up2 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=channels[3],
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        self.block2 = make_layer(
            block=InceptionResNet,
            block_layers=3, 
            channels=channels[3])

        self.conv2_cls = ME.MinkowskiConvolution(
            in_channels=channels[3],
            out_channels=1,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)
        self.pruning = ME.MinkowskiPruning()

    def prune_voxel(self, data, data_cls, nums, ground_truth, training):
        mask_topk = istopk(data_cls, nums)
        if training: 
            assert not ground_truth is None
            mask_true = isin(data_cls.C, ground_truth.C)
            mask = mask_topk + mask_true
        else: 
            mask = mask_topk
        data_pruned = self.pruning(data, mask.to(data.device))

        return data_pruned

    def forward(self, x, nums_list, ground_truth_list, training=True):
        #
        out = self.relu(self.conv0(self.relu(self.up0(x))))
        out = self.block0(out)
        out_cls_0 = self.conv0_cls(out)
        out = self.prune_voxel(out, out_cls_0, 
            nums_list[0], ground_truth_list[0], training)
        #
        out = self.relu(self.conv1(self.relu(self.up1(out))))
        out = self.block1(out)
        out_cls_1 = self.conv1_cls(out)
        out = self.prune_voxel(out, out_cls_1, 
            nums_list[1], ground_truth_list[1], training)
        #
        out = self.relu(self.conv2(self.relu(self.up2(out))))
        out = self.block2(out)
        out_cls_2 = self.conv2_cls(out)
        out = self.prune_voxel(out, out_cls_2, 
            nums_list[2], ground_truth_list[2], training)

        out_cls_list = [out_cls_0, out_cls_1, out_cls_2]

        return out_cls_list, out
