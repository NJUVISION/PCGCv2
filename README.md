# Multiscale Point Cloud Geometry Compression

​	We apply an **end-to-end learning framework** to compress the 3D  point cloud geometry (PCG) efficiently. Leveraging the sparsity nature of point cloud, we introduce the **multiscale structure** to represent native PCG compactly, offering the **hierarchical reconstruction** capability via progressive learnt re-sampling. Under this framework, we devise the **sparse convolution-based autoencoder** for feature analysis and aggregation. At the bottleneck layer, geometric occupancy information is losslessly encoded with a very small percentage of bits consumption, and corresponding feature attributes are lossy compressed. 

## News

- 2021.11.23 We proposed a better and unified PCGC framework based on PCGCv2, named **SparsePCGC**. It can support both **lossless** and lossy compression, as well as dense point clouds (e.g., 8iVFB) and sparse **LiDAR** point clouds (e.g., Ford). Here is the links: [paper](https://arxiv.org/abs/2111.10633) [code](https://github.com/NJUVISION/SparsePCGC)
- 2021.7.28 We have simplified the code, and use torchac to replace tensorflow-compression for arithmetic coding in the updated version. And the old version can be found [here](https://box.nju.edu.cn/f/60f21e96bdbe4e4d8208/).
- 2021.2.25 We have updated MinkowskiEngine to v0.5. The bug on GPU is fixed. And the encoding and decoding runtime is reduced.
- 2021.1.1 Our paper has been accepted by **DCC2021**! [[paper](https://arxiv.org/abs/2011.03799)]  [[presentation](https://sigport.org/documents/multiscale-point-cloud-geometry-compression)]




## Requirments
- python3.7 or 3.8
- cuda10.2 or 11.0
- pytorch1.7 or 1.8
- MinkowskiEngine 0.5 or higher (for sparse convolution)
- torchac 0.9.3 (for arithmetic coding) https://github.com/fab-jul/torchac
- tmc3 v12 (for lossless compression of downsampled point cloud coordinates) https://github.com/MPEGGroup/mpeg-pcc-tmc13

We recommend you to follow https://github.com/NVIDIA/MinkowskiEngine to setup the environment for sparse convolution. 

- Pretrained Models: https://box.nju.edu.cn/f/f44bb7ba98b149f898fb/
- Results (old version): https://box.nju.edu.cn/f/f2757a55e5e94440b2a7/
- Testdata: https://box.nju.edu.cn/f/a9e128f906ad4a738a3c/
- Training Dataset: https://box.nju.edu.cn/f/8a1fb24ce6d846fca722/

## Usage

### Testing
Please download the pretrained models and install tmc3 mentioned above first.
```shell
sudo chmod 777 tmc3 pc_error_d
python coder.py --filedir='longdress_vox10_1300.ply' --ckptdir='ckpts/r3_0.10bpp.pth' --scaling_factor=1.0 --rho=1.0 --res=1024
python test.py --filedir='longdress_vox10_1300.ply' --scaling_factor=1.0 --rho=1.0 --res=1024
python test.py --filedir='dancer_vox11_00000001.ply'--scaling_factor=1.0 --rho=1.0 --res=2048
python test.py --filedir='Staue_Klimt_vox12.ply' --scaling_factor=0.375 --rho=4.0 --res=4096
python test.py --filedir='House_without_roof_00057_vox12.ply' --scaling_factor=0.375 --rho=1.0 --res=4096
```
The testing rusults of 8iVFB can be found in `./results`

### Training
```shell
 python train.py --dataset='training_dataset_rootdir'
```


## Authors
These files are provided by Nanjing University  [Vision Lab](https://vision.nju.edu.cn/). And thanks for the help from Prof. Dandan Ding from Hangzhou Normal University and Prof. Zhu Li from University of Missouri at Kansas. Please contact us (mazhan@nju.edu.cn and wangjq@smail.nju.edu.cn) if you have any questions.
