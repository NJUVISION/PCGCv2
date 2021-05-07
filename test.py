import os

import numpy as np
import h5py
import open3d as o3d
import glob
import time
import subprocess
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from utils.gpcc_wrapper import load_ply_data, write_ply_data, gpcc_encode, gpcc_decode
from utils.pc_error_wrapper import pc_error

def kdtree_partition(pc, max_num):
    parts = []
    
    class KD_node:  
        def __init__(self, point=None, LL = None, RR = None):  
            self.point = point  
            self.left = LL  
            self.right = RR
            
    def createKDTree(root, data):
        if len(data) <= max_num:
            parts.append(data)
            return

        variances = (np.var(data[:, 0]), np.var(data[:, 1]), np.var(data[:, 2]))
        dim_index = variances.index(max(variances))
        data_sorted = data[np.lexsort(data.T[dim_index, None])]

        point = data_sorted[int(len(data)/2)]  
        root = KD_node(point)  
        root.left = createKDTree(root.left, data_sorted[: int((len(data) / 2))])  
        root.right = createKDTree(root.right, data_sorted[int((len(data) / 2)):]) 
        
        return root
    
    init_root = KD_node(None)
    root = createKDTree(init_root, pc)  

    return parts

def partition_point_cloud(filedir, bin_file, max_num=300000):
    filedirs = []
    pcd = o3d.io.read_point_cloud(filedir)
    # o3d.visualization.draw_geometries([pcd])
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    pc = np.hstack((points, colors))
    parts = kdtree_partition(pc, max_num)

    for i, pc_part in enumerate(parts):
        pcd_part = o3d.geometry.PointCloud()
        pcd_part.points = o3d.utility.Vector3dVector(pc_part[:,0:3])
        pcd_part.colors = o3d.utility.Vector3dVector(pc_part[:,3:6])

        filedir_part = os.path.dirname(bin_file)+'/'+os.path.split(filedir)[-1][:-4]+'_part'+str(i)+'.ply'
        o3d.io.write_point_cloud(filedir_part, pcd_part, write_ascii=True)
        filedirs.append(filedir_part)
    
    return filedirs, len(pc_part)

def load_sp_tensor(filedir, voxel_size=1, device='cpu'):
    pcd=o3d.io.read_point_cloud(filedir)

    if voxel_size == 1:
        downpcd = pcd
        #print(downpcd)
    else:
        # downpcd = pcd.voxel_down_sample(voxel_size= voxel_size)
        if o3d.__version__.split('.')[1] < '8': 
            downpcd = o3d.voxel_down_sample(pcd, voxel_size = voxel_size)
        else:
            downpcd = pcd.voxel_down_sample(voxel_size = voxel_size)
        #o3d.draw_geometries([downpcd])

        # Quantization
        points = np.asarray(downpcd.points)/voxel_size
        points = np.round(points)
        # colors = np.asarray(downpcd.colors)

        downpcd.points = o3d.utility.Vector3dVector(points)
        # downpcd.colors = o3d.utility.Vector3dVector(colors)
        #print('After DownSample:', downpcd)

    points = np.asarray(downpcd.points) 
    # colors = np.asarray(downpcd.colors)
    # Sparse Quantize
    
    feats = torch.ones(points.shape[0]).unsqueeze(-1)
    points, feats = ME.utils.sparse_quantize(coordinates=points, features=feats, quantization_size=1)
    coords, feats = ME.utils.sparse_collate([points], [feats])

    x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=device)
    
    return x

def sort_sparse_tensor(coords, feats):
    coords = coords.long()
    minimum = coords.min()
    coords = coords - minimum
    step = coords.max() + 1 
    coords_sum = coords[:, 1] + coords[:, 2]*step + coords[:, 3]*step*step
    ids = torch.argsort(coords_sum)
    coords_sum_sorted = coords_sum[ids]

    coords_sorted = torch.stack([(-minimum) * torch.ones(len(coords_sum_sorted)).to(minimum.device),
                                 coords_sum_sorted % step, 
                                 (coords_sum_sorted // step) % step, 
                                 (coords_sum_sorted // step // step) % step], 1)
    feats_sorted = feats[ids.long()]

    # feats_origin = feats_sorted[torch.argsort(ids)]
    # coords_origin = coords_sorted[torch.argsort(ids)]
    return coords_sorted.int()+minimum, feats_sorted, ids

def sort_xyz(coords):
    coords = coords.astype('int64')
    coords = torch.Tensor(coords)
    coords = coords.long()
    minimum = coords.min()
    # print(minimum)
    coords = coords - minimum
    step = coords.max() + 1
    # print(step)
    coords_sum = coords[:, 0] + coords[:, 1]*step + coords[:, 2]*step*step
    ids = torch.argsort(coords_sum)
    coords_sum_sorted = coords_sum[ids]

    coords_sorted = torch.stack([coords_sum_sorted % step, 
                                 (coords_sum_sorted // step) % step, 
                                 (coords_sum_sorted // step // step) % step], 1)

    return coords_sorted.int() + minimum

def encode(filedir, pcc, prefix, voxel_size):
    x = load_sp_tensor(filedir, voxel_size, device=device)
    # analysis transform
    with torch.no_grad():
        ys = pcc.encoder(x)
    y = ME.SparseTensor(ys[0].F, coordinates=ys[0].C, tensor_stride=8, device=device)
    # encode: y's coords 
    y_coords = (y.decomposed_coordinates[0]//y.tensor_stride[0]).cpu().numpy().astype('int')
    y_coords_name = prefix+'_coords.ply'
    write_ply_data(y_coords_name, y_coords)

    y_coords_binname = prefix+'_coords.bin'
    # gpcc_encode(plyname, binname, False)
    gpcc_encode(y_coords_name, y_coords_binname, False)
    os.system('rm '+y_coords_name)

    # encode: y's feats
    coords_sorted, feats_sorted, _ = sort_sparse_tensor(y.C, y.F)

    y_sorted = ME.SparseTensor(feats_sorted, 
                              coords_sorted,
                              tensor_stride=8, 
                              device=device)
    
    shape = y_sorted.F.shape
    strings, min_v, max_v = pcc.entropy_bottleneck.compress(y_sorted.F, device=device)
    target_label = [len(ys[1]), len(ys[2]), len(x)]
    
    y_feats_binname = prefix+'_feats.bin'
    with open(y_feats_binname, 'wb') as f:
        f.write(strings)
        
    head_binname = prefix+'_head.bin'
    with open(head_binname, 'wb') as f:
        f.write(np.array((min_v, max_v), dtype=np.int8).tobytes())
        f.write(np.array(shape, dtype=np.int32).tobytes())
        f.write(np.array(target_label, dtype=np.int32).tobytes())

    print('==========================', y_sorted.F.shape, y_sorted.C.shape)
    return y_coords_binname, y_feats_binname, head_binname

def decode(y_coords_binname, y_feats_binname, head_binname, pcc, rho, voxel_size):
    prefix = os.path.split(y_coords_binname)[-1].split('.')[0]
    # decode: y's coords
    y_coords_name = prefix+'_coords_rec.ply'
    # gpcc_decode(binname, plyname_rec, False)
    gpcc_decode(y_coords_binname, y_coords_name, False)

    y_coords = load_ply_data(y_coords_name)
    tensor_stride = 8
    y_coords = sort_xyz(y_coords)*tensor_stride
    
    # decode: y's feats
    with open(y_feats_binname, 'rb') as f:
        strings = f.read()

    with open(head_binname, 'rb') as f:
        min_v, max_v = np.frombuffer(f.read(1*2), dtype=np.int8)
        shape = np.frombuffer(f.read(4*2), dtype=np.int32)
        target_label = np.frombuffer(f.read(4*3), dtype=np.int32).tolist()
        
    y_feats = pcc.entropy_bottleneck.decompress(strings, min_v, max_v, shape, device=device)

    # print('---------------------', y_coords.shape, y_feats.shape)
    y_coords, y_feats = ME.utils.sparse_collate([y_coords], [y_feats])

    y = ME.SparseTensor(y_feats, 
                        y_coords,
                        tensor_stride=tensor_stride, 
                        device=device)

    # synthesis transform
    with torch.no_grad():
        out, out_cls, targets, keeps = pcc.decoder(y, target_label,                                                    
                                                   adaptive=True, 
                                                   rhos=[1.0, 1.0, rho], 
                                                   training=False)
    
    os.system('rm '+y_coords_name)
    os.system('rm '+y_coords_binname)
    os.system('rm '+y_feats_binname)
    os.system('rm '+head_binname)
    
    return out.decomposed_coordinates[0].cpu().numpy()*voxel_size

def partition_encode(partition_filedirs, pcc, max_num, voxel_size):  
    #partition to blocks
    coords_binname_set = []
    feats_binname_set = []
    head_binname_set = []
    
    for idx_part, partition_filedir in enumerate(partition_filedirs):
        prefix = os.path.split(partition_filedir)[-1].split('.')[0]
        coords_binname, feats_binname, head_binname = encode(partition_filedir, pcc, prefix, voxel_size)
        os.system('rm '+partition_filedir)
        coords_binname_set.append(coords_binname)
        feats_binname_set.append(feats_binname)
        head_binname_set.append(head_binname)

    return coords_binname_set, feats_binname_set, head_binname_set

def partition_decode(coords_binname_set, feats_binname_set, head_binname_set, pcc, rho, voxel_size):
    out_set = []
    for coords_binname, feats_binname, head_binname in zip(coords_binname_set, feats_binname_set, head_binname_set):
        out = decode(coords_binname, feats_binname, head_binname, pcc, rho, voxel_size)
        out_set.append(out)
    out = np.concatenate(out_set)

    return out

def write_file(prefix, binfile, num_parts):
    for idx_part in range(num_parts):
        coords_binname = prefix + '_part' + str(idx_part) + '_coords.bin'
        feats_binname = prefix + '_part' + str(idx_part) + '_feats.bin'
        head_binname  = prefix + '_part' + str(idx_part) + '_head.bin'

        # print(coords_binname, feats_binname, head_binname)

        coords_bin = open(coords_binname, 'rb').read()
        feats_bin = open(feats_binname, 'rb').read()
        head_bin = open(head_binname, 'rb').read()
        os.system('rm '+coords_binname)
        os.system('rm '+feats_binname)
        os.system('rm '+head_binname)

        binname = prefix+'_part' + str(idx_part) + '.bin'

        # print(binname)

        with open(binname, 'wb') as f:
            f.write(np.array((len(coords_bin), len(feats_bin), len(head_bin)), dtype=np.int32).tobytes())
            f.write(coords_bin)
            f.write(feats_bin)
            f.write(head_bin)


    lens_binname = []
    for idx_part in range(num_parts):
        binname = prefix+'_part' + str(idx_part) + '.bin'
        lens_binname.append(len(open(binname, 'rb').read()))

    all_binname = prefix+'.bin'

    with open(binfile, 'wb') as f:
        f.write(np.array(num_parts, dtype=np.uint8).tobytes())
        f.write(np.array(lens_binname, dtype=np.int32).tobytes())
        for idx_part in range(num_parts):
            binname = prefix+'_part' + str(idx_part) + '.bin'
            f.write(open(binname, 'rb').read())
            os.system('rm '+binname)
            
    return all_binname

def load_file(prefix):
    all_binname = prefix+'.bin'
    with open(all_binname, 'rb') as f:
        num_parts = np.frombuffer(f.read(1), dtype=np.uint8)
        num_parts = int(num_parts)
        lens_binname = np.frombuffer(f.read(4*num_parts), dtype=np.int32)

        for idx_part in range(num_parts):
            binname = prefix+'_part' + str(idx_part) + '.bin'
            strings = f.read(lens_binname[idx_part])
            with open(binname, 'wb') as f1:
                f1.write(strings)

    coords_binname_set = [] 
    feats_binname_set = []
    head_binname_set = []
    for idx_part in range(num_parts):
        
        binname = prefix+'_part' + str(idx_part) + '.bin'
        
        with open(binname, 'rb') as f:
            len_coords_bin, len_feats_bin, len_head_bin = np.frombuffer(f.read(4*3), dtype=np.int32)
        
            coords_bin = f.read(len_coords_bin)
            feats_bin = f.read(len_feats_bin)
            head_bin = f.read(len_head_bin)
            
            coords_binname = prefix + '_part' + str(idx_part) + '_coords.bin'
            feats_binname = prefix + '_part' + str(idx_part) + '_feats.bin'
            head_binname  = prefix + '_part' + str(idx_part) + '_head.bin'

            with open(coords_binname, 'wb') as f:
                f.write(coords_bin)
            with open(feats_binname, 'wb') as f:
                f.write(feats_bin)
            with open(head_binname, 'wb') as f:
                f.write(head_bin)
            coords_binname_set.append(coords_binname)
            feats_binname_set.append(feats_binname)
            head_binname_set.append(head_binname)
        
        os.system('rm '+binname)
    return coords_binname_set, feats_binname_set, head_binname_set

def metrics(filedir, out, coords_binname, feats_binname, head_binname, all_binname, resolution):
    rec_pcd = o3d.geometry.PointCloud()
    rec_pcd.points = o3d.utility.Vector3dVector(out)
    # rec_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=20))
    prefix = os.path.split(filedir)[-1].split('.')[0]
    recname = prefix+'_rec.ply'
    o3d.io.write_point_cloud(recname, rec_pcd, write_ascii=True)

    results = pc_error(filedir, recname, res=resolution, normal=True, show=False)

    # bpp
    ori_pcd = o3d.io.read_point_cloud(filedir)
    num_points = len(np.asarray(ori_pcd.points))

    if isinstance(coords_binname, list):
        coords_bytes = 0.
        for coords_bin in coords_binname:
            coords_bytes += os.path.getsize(coords_bin)
        coords_bpp = 8*coords_bytes / num_points
    else:
        coords_byte = os.path.getsize(coords_binname)
        coords_bpp = 8*coords_byte / num_points

    if isinstance(feats_binname, list):
        feats_bytes = 0.
        for feats_bin in feats_binname:
            feats_bytes += os.path.getsize(feats_bin)
        feats_bpp = 8*feats_bytes / num_points
    else:
        feats_byte = os.path.getsize(feats_binname)
        feats_bpp = 8*feats_byte / num_points
        
    if isinstance(head_binname, list):
        head_bytes = 0.
        for head_bin in head_binname:
            head_bytes += os.path.getsize(head_bin)
        head_bpp = 8*head_bytes / num_points
    else:
        head_byte = os.path.getsize(head_binname)
        head_bpp = 8*head_byte / num_points
    
    all_byte = os.path.getsize(all_binname)
    all_bpp = 8*all_byte / num_points

    results["n_points_ori"] = num_points
    results["n_points"] = len(out)
    results["bpp_coords"] = round(coords_bpp, 6)
    results["bpp_feats"] = round(feats_bpp, 6)
    results["bpp_head"] = round(head_bpp, 6)
    #results["bpp"] = round(coords_bpp + feats_bpp + head_bpp, 6)
    results["bpp"] = round(all_bpp, 6)
    
    return results

def compress(pc_file, binfile, ckptdir, voxel_size, max_num):
    if os.path.exists(ckptdir):
        ckpt = torch.load(ckptdir)
        pcc.encoder.load_state_dict(ckpt['encoder'])
        pcc.decoder.load_state_dict(ckpt['decoder'])
        pcc.entropy_bottleneck.load_state_dict(ckpt['entropy_bottleneck'])
    else:
        print('load failed!')
        return
    #
    #prefix = os.path.split(filedir)[-1].split('.')[0] + '_R' + str(idx_ckpt)
    prefix = os.path.split(pc_file)[-1].split('.')[0]
    
    #coords_binname, feats_binname, head_binname = encode(filedir, pcc, prefix)
    partition_filedirs, partition_num = partition_point_cloud(pc_file, binfile, max_num)
    coords_binname, feats_binname, head_binname = partition_encode(partition_filedirs, pcc, max_num, voxel_size)
    all_binname = write_file(prefix, binfile, len(partition_filedirs))
    
def decompress(binfile, pcfile, ckptdir, voxel_size, rho):
    if os.path.exists(ckptdir):
        ckpt = torch.load(ckptdir)
        pcc.encoder.load_state_dict(ckpt['encoder'])
        pcc.decoder.load_state_dict(ckpt['decoder'])
        pcc.entropy_bottleneck.load_state_dict(ckpt['entropy_bottleneck'])
    else:
        print('load failed!')
        return
    
    prefix = os.path.splitext(binfile)[0]
    coords_binname, feats_binname, head_binname = load_file(prefix)
    #out = decode(coords_binname, feats_binname, head_binname, pcc, rho)
    out = partition_decode(coords_binname, feats_binname, head_binname, pcc, rho=rho, voxel_size=voxel_size)

    # save dec files
    rec_pcd = o3d.geometry.PointCloud()
    rec_pcd.points = o3d.utility.Vector3dVector(out)
    o3d.io.write_point_cloud(pcfile, rec_pcd, write_ascii=True)

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("command", choices=["compress", "decompress"],
                        help="What to do: 'compress' reads a point cloud (.ply format) "
                            "and writes compressed binary files. 'decompress' "
                            "reads binary files and reconstructs the point cloud (.ply format). "
                            "input and output filenames need to be provided for the latter. ")
    parser.add_argument("input", nargs="?", help="Input filepath.")
    parser.add_argument("output", nargs="?", help="Output filepath.")
    parser.add_argument("--ckptdir", type=str, default='./ckpts/c8_a10_32000.pth', help="ckptdir")
    parser.add_argument("--voxel_size", type=int, default=1, help="voxel_size")
    parser.add_argument("--max_num", type=int, default=2e6, help="max number of points")
    parser.add_argument("--rho", type=float, default=1, help="output_num/input_num")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    from PCCModel import PCC
    pcc = PCC(channels=8).to(device)
    
    args = parse_args()
    
    if args.command == 'compress':
        compress(args.input, args.output, args.ckptdir, args.voxel_size, args.max_num)
    elif args.command == 'decompress':
        decompress(args.input, args.output, args.ckptdir, args.voxel_size, args.rho)