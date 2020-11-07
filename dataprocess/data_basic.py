# Copyright (c) Nanjing University, Vision Lab.
# Jianqiang Wang (wangjq@smail.nju.edu.cn), Zhan Ma (mazhan@nju.edu.cn); Nanjing University, Vision Lab.
# Last update: 2020.06.06

#import open3d as o3d
import numpy as np
import h5py
import torch
import MinkowskiEngine as ME
import random

def loadh5(filedir, color_format='rgb'):
  """Load coords & feats from h5 file.

  Arguments: file direction

  Returns: coords & feats.
  """
  pc = h5py.File(filedir, 'r')['data'][:]

  coords = pc[:,0:3].astype('int32')

  if color_format == 'rgb':
    feats = pc[:,3:6]/255. 
  elif color_format == 'yuv':
    R, G, B = pc[:, 3:4], pc[:, 4:5], pc[:, 5:6]
    Y = 0.257*R + 0.504*G + 0.098*B + 16
    Cb = -0.148*R - 0.291*G + 0.439*B + 128
    Cr = 0.439*R - 0.368*G - 0.071*B + 128
    feats = np.concatenate((Y,Cb,Cr), -1)/256.
  elif color_format == 'geometry':
    feats = np.expand_dims(np.ones(coords.shape[0]), 1)
    
  feats = feats.astype('float32')

  return coords, feats

def loadply(filedir, color_format='rgb'):
  """Load coords & feats from ply file.
  
  Arguments: file direction.
  
  Returns: coords & feats.
  """

  files = open(filedir)
  coords = []
  feats = []
  for line in files:
    wordslist = line.split(' ')
    try:
      x, y, z, r, g, b = float(wordslist[0]),float(wordslist[1]),float(wordslist[2]), \
      float(wordslist[3]),float(wordslist[4]),float(wordslist[5])
    except ValueError:
      continue
    coords.append([x,y,z])
    feats.append([r,g,b])

  coords = np.array(coords).astype('int32')
  feats = np.array(feats).astype('float32')
  
  if color_format == 'rgb':
    feats = feats/255.
  elif color_format == 'yuv':
    R, G, B = feats[:, 0:1], feats[:, 1:2], feats[:, 2:3]
    Y = 0.257*R + 0.504*G + 0.098*B + 16
    Cb = -0.148*R - 0.291*G + 0.439*B + 128
    Cr = 0.439*R - 0.368*G - 0.071*B + 128
    feats = np.concatenate((Y,Cb,Cr), -1)/256.
    
  elif color_format=='geometry':
    feats = np.expand_dims(np.ones(coords.shape[0]), 1)
  
  feats = feats.astype('float32')

  coords, feats = ME.utils.sparse_quantize(coords=coords, feats=feats, quantization_size=1)

  feats = feats.astype('float32')

  return coords, feats

def data_loader(filedirs, batch_size, color_format='rgb'):
  """
  Load batch of coordinates & attribute(color).
  
  Args:
      filedirs: strings list
      batch size: uint  
      
  Returns:
      coords: shape=[N, 3+1], Type=Tensor
      feats:  shape=[N, 3], Type=Tensor
  """

  filedir_batch = random.sample(filedirs, batch_size)
  
  coords_batch = []
  feats_batch = []

  for _, filedir in enumerate(filedir_batch):
    if filedir.endswith('.h5'):
      coords, feats = loadh5(filedir, color_format)
    elif filedir.endswith('.ply'):
      coords, feats = loadply(filedir, color_format)
    
    coords_batch.append(coords)
    feats_batch.append(feats)

  return coords_batch, feats_batch




