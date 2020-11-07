# Copyright (c) Nanjing University, Vision Lab.
# Jianqiang Wang (wangjq@smail.nju.edu.cn), Zhan Ma (mazhan@nju.edu.cn); Nanjing University, Vision Lab.
# Last update: 2020.06.06

import os
import sys
# sys.path.append('.')
import glob
import subprocess
import argparse
import logging
from time import time
import torch
import torch.utils.data
from torch.utils.data.sampler import Sampler

import numpy as np
import MinkowskiEngine as ME

from dataprocess.data_basic import loadh5, loadply


class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)

def collate_pointcloud_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1

    list_data = new_list_data

    if len(list_data) == 0:
        raise ValueError('No data in the batch')

    # coords, feats, labels = list(zip(*list_data))
    coords, feats = list(zip(*list_data))

    coords_batch = ME.utils.batched_coordinates(coords)
    feats_batch = torch.from_numpy(np.vstack(feats)).float()

    return coords_batch, feats_batch


class PCDataset(torch.utils.data.Dataset):

    def __init__(self, files, feature_format):
        self.files = []
        self.cache = {}
        self.last_cache_percent = 0
        self.files = files
        self.feature_format = feature_format

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pc_file = self.files[idx]

        if idx in self.cache:
            coords, feats = self.cache[idx]
        else:
            if pc_file.endswith('.h5'):
              coords, feats = loadh5(pc_file, self.feature_format)
            elif pc_file.endswith('.ply'):
              coords, feats = loadply(pc_file, self.feature_format)

            self.cache[idx] = (coords, feats)

            cache_percent = int((len(self.cache) / len(self)) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                print('cache percent:', len(self.cache) / len(self))
                self.last_cache_percent = cache_percent

        return (coords, feats)


def make_data_loader(dataset, batch_size, shuffle, num_workers, repeat):
    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_pointcloud_fn, 
        'pin_memory': True,
        'drop_last': False
    }

    if repeat:
        args['sampler'] = InfSampler(dataset, shuffle)
    else:
        args['shuffle'] = shuffle

    loader = torch.utils.data.DataLoader(dataset, **args)

    return loader


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default='')
    args = parser.parse_args()

    import glob
    filedirs = glob.glob(args.dataset+'*.h5')

    test_dataset = PCDataset(filedirs[:100], feature_format='rgb')
    test_dataloader = make_data_loader(dataset=test_dataset, 
                                    batch_size=8, 
                                    shuffle=True, 
                                    num_workers=1,
                                    repeat=True)
    

    test_iter = iter(test_dataloader)

    import time
    s = time.time()

    for i in range(1000):
        coords, feats = test_iter.next()
        if i % 20 == 0:
            print('iter::：:', i, time.time() - s)
            s = time.time()


