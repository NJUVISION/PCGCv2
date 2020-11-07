import torch
import MinkowskiEngine as ME
import numpy as np

def _count_sparse_conv(kernel_size, in_channels, out_channels):
    total_params = pow(kernel_size[0], 3) * in_channels * out_channels + out_channels
    return total_params

def count_sparse_conv(m: ME.MinkowskiConvolution, x: ME.SparseTensor, y: ME.SparseTensor):
    total_params = _count_sparse_conv(m.kernel_size, m.in_channels, m.out_channels)
    n_points = len(y.C)
    m.total_params += torch.DoubleTensor([int(total_params)])
    # print(np.int64(total_params) * np.int64(n_points)/pow(10,9))
    m.total_ops += torch.LongTensor([np.int64(total_params) * np.int64(n_points)])

def _count_sparse_deconv(kernel_size, in_channels, out_channels):
    total_params = pow(kernel_size[0], 3) * in_channels * out_channels + out_channels
    return total_params

def count_sparse_deconv(m: ME.MinkowskiConvolutionTranspose, x: ME.SparseTensor, y: ME.SparseTensor):
    total_params = _count_sparse_deconv(m.kernel_size, m.in_channels, m.out_channels)
    n_points = len(y.C)
    m.total_params += torch.DoubleTensor([int(total_params)])
    # print(m, np.int64(total_params) * np.int64(n_points)/pow(10,9))
    m.total_ops += torch.LongTensor([np.int64(total_params) * np.int64(n_points)])

