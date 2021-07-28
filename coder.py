import os, time
import numpy as np
import torch
import MinkowskiEngine as ME
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from data_utils import array2vector, istopk, sort_spare_tensor, load_sparse_tensor
from data_utils import write_ply_ascii_geo, read_ply_ascii_geo

from gpcc import gpcc_encode, gpcc_decode
from pc_error import pc_error

from pcc_model import PCCModel


class CoordinateCoder():
    """encode/decode coordinates using gpcc
    """
    def __init__(self, filename):
        self.filename = filename
        self.ply_filename = filename + '.ply'

    def encode(self, coords, postfix=''):
        coords = coords.numpy().astype('int')
        write_ply_ascii_geo(filedir=self.ply_filename, coords=coords)
        gpcc_encode(self.ply_filename, self.filename+postfix+'_C.bin')
        os.system('rm '+self.ply_filename)
        
        return 

    def decode(self, postfix=''):
        gpcc_decode(self.filename+postfix+'_C.bin', self.ply_filename)
        coords = read_ply_ascii_geo(self.ply_filename)
        os.system('rm '+self.ply_filename)
        
        return coords


class FeatureCoder():
    """encode/decode feature using learned entropy model
    """
    def __init__(self, filename, entropy_model):
        self.filename = filename
        self.entropy_model = entropy_model.cpu()

    def encode(self, feats, postfix=''):
        strings, min_v, max_v = self.entropy_model.compress(feats.cpu())
        shape = feats.shape
        with open(self.filename+postfix+'_F.bin', 'wb') as fout:
            fout.write(strings)
        with open(self.filename+postfix+'_H.bin', 'wb') as fout:
            fout.write(np.array(shape, dtype=np.int32).tobytes())
            fout.write(np.array(len(min_v), dtype=np.int8).tobytes())
            fout.write(np.array(min_v, dtype=np.float32).tobytes())
            fout.write(np.array(max_v, dtype=np.float32).tobytes())
            
        return 

    def decode(self, postfix=''):
        with open(self.filename+postfix+'_F.bin', 'rb') as fin:
            strings = fin.read()
        with open(self.filename+postfix+'_H.bin', 'rb') as fin:
            shape = np.frombuffer(fin.read(4*2), dtype=np.int32)
            len_min_v = np.frombuffer(fin.read(1), dtype=np.int8)[0]
            min_v = np.frombuffer(fin.read(4*len_min_v), dtype=np.float32)[0]
            max_v = np.frombuffer(fin.read(4*len_min_v), dtype=np.float32)[0]
            
        feats = self.entropy_model.decompress(strings, min_v, max_v, shape, channels=shape[-1])
        
        return feats


class Coder():
    def __init__(self, model, filename):
        self.model = model 
        self.filename = filename
        self.coordinate_coder = CoordinateCoder(filename)
        self.feature_coder = FeatureCoder(self.filename, model.entropy_bottleneck)

    @torch.no_grad()
    def encode(self, x, postfix=''):
        # Encoder
        y_list = self.model.encoder(x)
        y = sort_spare_tensor(y_list[0])
        num_points = [len(ground_truth) for ground_truth in y_list[1:] + [x]]
        with open(self.filename+postfix+'_num_points.bin', 'wb') as f:
            f.write(np.array(num_points, dtype=np.int32).tobytes())
        self.feature_coder.encode(y.F, postfix=postfix)
        self.coordinate_coder.encode((y.C//y.tensor_stride[0]).detach().cpu()[:,1:], postfix=postfix)
        
        return y

    @torch.no_grad()
    def decode(self, postfix=''):
        # decode coords
        y_C = self.coordinate_coder.decode(postfix=postfix)
        y_C = torch.cat((torch.zeros((len(y_C),1)).int(), torch.tensor(y_C).int()), dim=-1)
        indices_sort = np.argsort(array2vector(y_C, y_C.max()+1))
        y_C = y_C[indices_sort]
        # decode feat
        y_F = self.feature_coder.decode(postfix=postfix)
        y = ME.SparseTensor(features=y_F, coordinates=y_C*8,
                            tensor_stride=8, device=device)
        # decode label
        with open(self.filename+postfix+'_num_points.bin', 'rb') as fin:
            num_points = np.frombuffer(fin.read(4*3), dtype=np.int32).tolist()
            num_points = [[num] for num in num_points]
        # decode
        _, out = self.model.decoder(y, nums_list=num_points, ground_truth_list=[None]*3, training=False)

        return out

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ckptdir", default='ckpts/r3_0.10bpp.pth')
    parser.add_argument("--filedir", default='../../../testdata/8iVFB/longdress_vox10_1300.ply')
    args = parser.parse_args()
    filedir = args.filedir

    # load data
    start_time = time.time()
    input_data = load_sparse_tensor(filedir, device)
    print('Loading Time:\t', round(time.time() - start_time, 4), 's')

    outdir = './output'
    if not os.path.exists(outdir): os.makedirs(outdir)
    filename = os.path.split(filedir)[-1].split('.')[0]
    filename = os.path.join(outdir, filename)
    print(filename)
    x = sort_spare_tensor(input_data)

    # model
    print('='*10, 'Test', '='*10)
    model = PCCModel().to(device)
    if os.path.exists(args.ckptdir):
        ckpt = torch.load(args.ckptdir)
        model.load_state_dict(ckpt['model'])
        print('load from ', args.ckptdir)
    else: print('load failed!')

    # coder
    coder = Coder(model=model, filename=filename)

    # encode
    start_time = time.time()
    _ = coder.encode(x)
    print('Enc Time:', round(time.time() - start_time, 3), 's')

    # decode
    start_time = time.time()
    x_dec = coder.decode()
    print('Dec Time:', round(time.time() - start_time, 3), 's')

    # bitrate
    bits = np.array([os.path.getsize(filename + postfix)*8 \
                            for postfix in ['_C.bin', '_F.bin', '_H.bin', '_num_points.bin']])
    bpp = (bits/len(x)).round(3)
    print(bits, 'bits', '\t', bpp, 'bpp')
    print(sum(bits), 'bits', '\t',  sum(bpp).round(3), 'bpp')

    # distortion
    write_ply_ascii_geo(filename+'.ply', x.C.detach().cpu().numpy()[:,1:])
    write_ply_ascii_geo(filename+'_dec.ply', x_dec.C.detach().cpu().numpy()[:,1:])
    pc_error_metrics = pc_error(filename+'.ply', filename+'_dec.ply', res=1024, show=False)
    print('pc_error_metrics:', pc_error_metrics)
    print('D1 PSNR', pc_error_metrics["mseF,PSNR (p2point)"][0])