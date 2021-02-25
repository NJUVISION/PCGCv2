# Copyright (c) Nanjing University, Vision Lab.
# Jianqiang Wang (wangjq@smail.nju.edu.cn), Zhan Ma (mazhan@nju.edu.cn); Nanjing University, Vision Lab.
# Last update: 2020.06.06

import numpy as np
import h5py
import os, sys
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF

from models.BasicBlock import ResNet, InceptionResNet

import time

class Encoder(nn.Module):
    """
    Encoder
    """

    def __init__(self, channels, block_layers, block):
        nn.Module.__init__(self)
        in_nchannels=1
        ch = [16, 32, 64, 32, channels]
        if block == 'ResNet':
            self.block = ResNet
        elif block == 'InceptionResNet':
            self.block = InceptionResNet


        self.conv0 = ME.MinkowskiConvolution(
            in_channels=in_nchannels,
            out_channels=ch[0],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down0 = ME.MinkowskiConvolution(
            in_channels=ch[0],
            out_channels=ch[1],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.block0 = self.make_layer(
            self.block, block_layers, ch[1])        

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=ch[1],
            out_channels=ch[1],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down1 = ME.MinkowskiConvolution(
            in_channels=ch[1],
            out_channels=ch[2],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.block1 = self.make_layer(
            self.block, block_layers, ch[2])

        self.conv2 = ME.MinkowskiConvolution(
            in_channels=ch[2],
            out_channels=ch[2],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down2 = ME.MinkowskiConvolution(
            in_channels=ch[2],
            out_channels=ch[3],
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.block2 = self.make_layer(
            self.block, block_layers, ch[3])

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=ch[3],
            out_channels=ch[4],
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)

    def make_layer(self, block, block_layers, channels):
        layers = []
        for i in range(block_layers):
            layers.append(block(channels=channels))
            
        return nn.Sequential(*layers)


    def forward(self, x):
        out0 = self.relu(self.down0(self.relu(self.conv0(x))))
        out0 = self.block0(out0)
        out1 = self.relu(self.down1(self.relu(self.conv1(out0))))
        out1 = self.block1(out1)
        out2 = self.relu(self.down2(self.relu(self.conv2(out1))))
        out2 = self.block2(out2)
        out2 = self.conv3(out2)

        return [out2, out1, out0]

class Decoder(nn.Module):
    """
    Decoder
    """

    def __init__(self, channels, block_layers, block):
        nn.Module.__init__(self)
        out_nchannel=1
        ch = [channels, 64, 32, 16]
        if block == 'ResNet':
            self.block = ResNet
        elif block == 'InceptionResNet':
            self.block = InceptionResNet

        self.up0 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=ch[0],
            out_channels=ch[1],
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3)
        self.conv0 = ME.MinkowskiConvolution(
            in_channels=ch[1],
            out_channels=ch[1],
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        self.block0 = self.make_layer(
            self.block, block_layers, ch[1])

        self.conv0_cls = ME.MinkowskiConvolution(
            in_channels=ch[1],
            out_channels=out_nchannel,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)

        self.up1 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=ch[1],
            out_channels=ch[2],
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3)
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=ch[2],
            out_channels=ch[2],
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        self.block1 = self.make_layer(
            self.block, block_layers, ch[2])

        self.conv1_cls = ME.MinkowskiConvolution(
            in_channels=ch[2],
            out_channels=out_nchannel,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)

        self.up2 = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=ch[2],
            out_channels=ch[3],
            kernel_size= 2,
            stride=2,
            bias=True,
            dimension=3)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=ch[3],
            out_channels=ch[3],
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)
        self.block2 = self.make_layer(
            self.block, block_layers, ch[3])

        self.conv2_cls = ME.MinkowskiConvolution(
            in_channels=ch[3],
            out_channels=out_nchannel,
            kernel_size= 3,
            stride=1,
            bias=True,
            dimension=3)

        self.relu = ME.MinkowskiReLU(inplace=True)
        # self.relu = ME.MinkowskiELU(inplace=True)

        # pruning
        self.pruning = ME.MinkowskiPruning()

    def make_layer(self, block, block_layers, channels):
        layers = []
        for i in range(block_layers):
            layers.append(block(channels=channels))
            
        return nn.Sequential(*layers)

    # get target from label key or sparse tensor.
    def get_target_by_key(self, out, target_key):

        with torch.no_grad():
            target = torch.zeros(len(out), dtype=torch.bool)
            cm = out.coords_man
            strided_target_key = cm.stride(
                target_key, out.tensor_stride[0], force_creation=True)
            # kernel size = 1
            ins, outs = cm.get_kernel_map(
                out.coords_key,
                strided_target_key,
                kernel_size=1,
                region_type=1)

            for curr_in in ins:
                target[curr_in] = 1
        return target.bool()

    def get_target_by_sp_tensor(self, out, target_sp_tensor):

        with torch.no_grad():
            def ravel_multi_index(coords, step):
                coords = coords.long()
                step = step.long()
                coords_sum = coords[:, 0] \
                          + coords[:, 1]*step \
                          + coords[:, 2]*step*step \
                          + coords[:, 3]*step*step*step
                return coords_sum

            step = max(out.C.max(), target_sp_tensor.C.max()) + 1

            out_sp_tensor_coords_1d = ravel_multi_index(out.C, step)
            in_sp_tensor_coords_1d = ravel_multi_index(target_sp_tensor.C, step)

            # test whether each element of a 1-D array is also present in a second array.
            target = np.in1d(out_sp_tensor_coords_1d.cpu().numpy(), 
                            in_sp_tensor_coords_1d.cpu().numpy())

        return torch.Tensor(target).bool()

    def get_coords_nums_by_key(self, out, target_key):
        with torch.no_grad():
            cm = out.coords_man
            strided_target_key = cm.stride(target_key, out.tensor_stride[0], force_creation=True)

            ins, outs = cm.get_kernel_map(
                out.coords_key,
                strided_target_key,
                kernel_size=1,
                region_type=1)
            
            row_indices_per_batch = cm.get_row_indices_per_batch(out.coords_key)
            
            coords_nums = [len(np.in1d(row_indices,ins[0]).nonzero()[0]) for _, row_indices in enumerate(row_indices_per_batch)]
            # coords_nums = [len(np.intersect1d(row_indices,ins[0])) for _, row_indices in enumerate(row_indices_per_batch)]
        
        return coords_nums

    def keep_adaptive(self, out, coords_nums, rho=1.0):

        with torch.no_grad():
            keep = torch.zeros(len(out), dtype=torch.bool)
            #  get row indices per batch.
            # row_indices_per_batch = out.coords_man.get_row_indices_per_batch(out.coords_key)
            row_indices_per_batch = out._batchwise_row_indices

            for row_indices, ori_coords_num in zip(row_indices_per_batch, coords_nums):
                coords_num = min(len(row_indices), ori_coords_num*rho)# select top k points.
                values, indices = torch.topk(out.F[row_indices].squeeze(), int(coords_num))
                keep[row_indices[indices]]=True

        return keep


    def forward(self, x, target_label, adaptive, rhos=[1.0, 1.0, 1.0], training=True):
        if isinstance(target_label, ME.CoordinateMapKey):
            target_format = 'key'
        elif isinstance(target_label, list):
            if isinstance(target_label[0], ME.SparseTensor):
                target_format = 'sp_tensor'
            elif isinstance(target_label[0], int):
                target_format = 'num'
        else:
            print('Target Label Format Error!')
            sys.exit(0)
        targets = []
        out_cls = []
        keeps = []

        # Decode 0.
        out0 = self.relu(self.conv0(self.relu(self.up0(x))))
        out0 = self.block0(out0)
        out0_cls = self.conv0_cls(out0)

        # get target 0.
        if target_format == 'key':
            target0 = self.get_target_by_key(out0, target_label)
        elif target_format == 'sp_tensor':
            target0 = self.get_target_by_sp_tensor(out0, target_label[0])
        elif target_format == 'num':
            target0 = target_label[0]

        targets.append(target0)
        out_cls.append(out0_cls)

        # get keep 0.
        if adaptive:
            if target_format == 'key':
                coords_nums0 = self.get_coords_nums_by_key(out0, target_label)
            elif target_format == 'sp_tensor':
                coords_nums0 = [len(coords) for coords in target_label[0].decomposed_coordinates]
            elif target_format == 'num':
                coords_nums0 = [target_label[0]]

            keep0 = self.keep_adaptive(out0_cls, coords_nums0, rho=rhos[0])
        else:
            keep0 = (out0_cls.F > 0).cpu().squeeze()
            if out0_cls.F.max() < 0:
                # keep at least one points.
                print('===0; max value < 0', out0_cls.F.max())
                _, idx = torch.topk(out0_cls.F.squeeze(), 1)
                keep0[idx] = True

        keeps.append(keep0)

        # If training, force target shape generation, use net.eval() to disable
        if training:
            keep0 += target0


        # Remove voxels
        out0_pruned = self.pruning(out0, keep0.to(out0.device))

        # Decode 1.
        out1 = self.relu(self.conv1(self.relu(self.up1(out0_pruned))))
        out1 = self.block1(out1)
        out1_cls = self.conv1_cls(out1)

        # get target 1.
        if target_format == 'key':
            target1 = self.get_target_by_key(out1, target_label)
        elif target_format == 'sp_tensor':
            target1 = self.get_target_by_sp_tensor(out1, target_label[1])
        elif target_format == 'num':
            target1 = target_label[1]

        targets.append(target1)
        out_cls.append(out1_cls)

        # get keep 1.
        if adaptive:
            if target_format == 'key':
                coords_nums1 = self.get_coords_nums_by_key(out1, target_label)
            elif target_format == 'sp_tensor':
                coords_nums1 = [len(coords) for coords in target_label[1].decomposed_coordinates]
            elif target_format == 'num':
                coords_nums1 = [target_label[1]]

            keep1 = self.keep_adaptive(out1_cls, coords_nums1, rho=rhos[1])
        else:
            keep1 = (out1_cls.F > 0).cpu().squeeze()
            if out1_cls.F.max() < 0:
                # keep at least one points.
                print('===1; max value < 0', out1_cls.F.max())
                _, idx = torch.topk(out1_cls.F.squeeze(), 1)
                keep1[idx] = True

        keeps.append(keep1)

        if training:
            keep1 += target1
            
        # Remove voxels
        out1_pruned = self.pruning(out1, keep1.to(out1.device))

        # Decode 2.
        out2 = self.relu(self.conv2(self.relu(self.up2(out1_pruned))))
        out2 = self.block2(out2)
        out2_cls = self.conv2_cls(out2)

        # get target 2.
        if target_format == 'key':
            target2 = self.get_target_by_key(out2, target_label)
        elif target_format == 'sp_tensor':
            target2 = self.get_target_by_sp_tensor(out2, target_label[2])
        elif target_format == 'num':
            target2 = target_label[2]

        targets.append(target2)
        out_cls.append(out2_cls)

        # get keep 2.
        if adaptive:
            if target_format == 'key':
                coords_nums2 = self.get_coords_nums_by_key(out2, target_label)
            elif target_format == 'sp_tensor':
                coords_nums2 = [len(coords) for coords in target_label[2].decomposed_coordinates]
            elif target_format == 'num':
                coords_nums2 = [target_label[2]]

            keep2 = self.keep_adaptive(out2_cls, coords_nums2, rho=rhos[2])
        else:
            keep2 = (out2_cls.F > 0).cpu().squeeze()
            if out2_cls.F.max() < 0:
                # keep at least one points.
                print('===2; max value < 0', out2_cls.F.max())
                _, idx = torch.topk(out2_cls.F.squeeze(), 1)
                keep2[idx] = True

        keeps.append(keep2)
        
        # Remove voxels
        out2_pruned = self.pruning(out2_cls, keep2.to(out2_cls.device))

        return out2_pruned, out_cls, targets, keeps


if __name__ == '__main__':
    encoder = Encoder(8)
    print(encoder)
    decoder = Decoder(8)
    print(decoder)
