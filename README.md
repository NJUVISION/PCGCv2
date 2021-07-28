# Multiscale Point Cloud Geometry Compression

​	We apply an **end-to-end learning framework** to compress the 3D  point cloud geometry (PCG) efficiently. Leveraging the sparsity nature of point cloud, we introduce the **multiscale structure** to represent native PCG compactly, offering the **hierarchical reconstruction** capability via progressive learnt re-sampling. Under this framework, we devise the **sparse convolution-based autoencoder** for feature analysis and aggregation. At the bottleneck layer, geometric occupancy information is losslessly encoded with a very small percentage of bits consumption, and corresponding feature attributes are lossy compressed. 

## News

- 2021.1.1 Our paper has been accepted by **DCC2021**! [[paper](https://arxiv.org/abs/2011.03799)]  [[presentation](https://sigport.org/documents/multiscale-point-cloud-geometry-compression)]
- 2021.2.25 We have updated MinkowskiEngine to v0.5. The bug on GPU is fixed. And the encoding and decoding runtime is reduced.
- 2021.7.28 We have simplified the code, and use torchac to replace tensorflow-compression for arithmetic coding in the updated version. And the old version can be found [here](https://box.nju.edu.cn/f/60f21e96bdbe4e4d8208/).


## Requirments
- python3.7 or 3.8
- cuda10.2 or 11.0
- pytorch1.7 or 1.8
- MinkowskiEngine 0.5 or higher (for sparse convolution)
- torchac 0.9.3 (for arithmetic coding) https://github.com/fab-jul/torchac

We recommend you to follow https://github.com/NVIDIA/MinkowskiEngine to setup the environment for sparse convolution. 

- Pretrained Models: https://box.nju.edu.cn/f/46d9206c6565471fb256/
- Results: https://box.nju.edu.cn/f/f2757a55e5e94440b2a7/
- Testdata: https://box.nju.edu.cn/f/e7a4578decf24cfa8e09/
- Training Dataset: http://yun.nju.edu.cn/f/7c81b0e501/

## Usage

### Training
```shell
 python train.py --dataset='training_dataset_rootdir'
```

### Testing
```shell
sudo chmod 777 tmc3 pc_error_d
python coder.py --ckptdir='ckpts/r3_0.10bpp.pth' --filedir='longdress_vox10_1300.ply'
```

## Authors
These files are provided by Nanjing University  [Vision Lab](https://vision.nju.edu.cn/). And thanks for the help from Prof. Dandan Ding from Hangzhou Normal University and Prof. Zhu Li from University of Missouri at Kansas. Please contact us (mazhan@nju.edu.cn and wangjq@smail.nju.edu.cn) if you have any questions.