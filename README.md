# Multiscale Point Cloud Geometry Compression

​	We apply an **end-to-end learning framework** to compress the 3D  point cloud geometry (PCG) efficiently. Leveraging the sparsity nature of point cloud, we introduce the **multiscale structure** to represent native PCG compactly, offering the **hierarchical reconstruction** capability via progressive learnt re-sampling. Under this framework, we devise the **sparse convolution-based autoencoder** for feature analysis and aggregation. At the bottleneck layer, geometric occupancy information is losslessly encoded with a very small percentage of bits consumption, and corresponding feature attributes are lossy compressed. 

## Framework

<p align="center">
  <img src="figs/framework.png?raw=true" alt="introduction" width="800"/> 

  **Multiscale PCG Compression:**  (a) "Conv c*n^3" denotes the sparse convolution with 'c' output channels and n^3 kernel size, "Q" stands for Quantization, "AE" and "AD" are Arithmetic Encoder and Decoder respectively. "IRN" is Inception-Residual Network used for efficient feature aggregation. (a) network structure of IRN unit, (c) adaptive contexts conditioned on autoregressive priors.
  <img src="figs/reconstruct.png?raw=true" alt="introduction" width="800"/> 

​	**Binary classification based hierarchical reconstruction**: The top part shows the encoding process: **(a), (b), (c), (d)** are exemplified from a size of 32^3 to  4^3, by halving each geometric dimension scale step-by-step; The bottom part illustrates corresponding  hierarchical reconstructions, geometric models are upscaled and classified gradually from the rightmost to the leftmost position: **(e), (g), (i)** are convolutionally upscaled from lower scales with size of  8^3, 16^3 and 32^3. Different colors are used to differentiate the **probabilities of voxel-being-occupied** (i.e., the greener, the closer  to 1, and the bluer, the closer to 0); and **(f), (h), (j)** are the reconstructions after classification with green blocks for true classified voxels,  blue for false positive, and red for false negative voxels.
</p>

## Requirments

- python3.6 or higher
- cuda10.1 or higher
- pytorch1.3 or higher
- MinkowskiEngine 0.4 or higher (for sparse convolution)
- tensorflow1.13 (for Arithmetic Encoder)

We recommend you to follow the [MinkowskiEngine installation instruction](https://github.com/stanfordvl/MinkowskiEngine) to setup the environment for the sparse convolution. 
We also provide docker image for direct usage. (nvidia driver >= 418)

```shell
wget -O pytorch13me041.tar http://yun.nju.edu.cn/f/7b718f2c21/?raw=1
docker load --input pytorch13me041.tar
```
then you can open a container and run the code:
```shell
nvidia-docker run --shm-size=4g --rm -it --name learnedpcc -v /home/ubuntu:/home/ubuntu pytorch/pytorch:1.3-me0.4  bash

```

- Pretrained Models: http://yun.nju.edu.cn/f/11b6942485/
- Testdata: http://yun.nju.edu.cn/f/76beec38bc/
- Training Dataset: http://yun.nju.edu.cn/f/7c81b0e501/

## Usage

### Training
```shell
 python train.py --dataset='training_dataset/' --dataset_8i = 'testdata/8iVFB/'
```

### Testing
```shell
python eval.py --filedir='testdata/8iVFB/redandblack_vox10_1550_n.ply' --ckptdir='ckpts/c8_a2_32000.pth'
```

or test all data
```shell
python eval.py --test_all
```
### Examples
`demo.ipynb`


## Comparison
### Objective Comparison
See `results.ipynb`
<p align="center">
  <img src="figs/rdcurve.png?raw=true" alt="" width="800"/>
 </p>  
 
### Qualitative Evaluation
<p align="center">
  <img src="figs/vis.png?raw=true" alt="introduction" width="800"/>
 </p>  

## Update
- 2020.06 paper submission.
- 2020.10.29 open source code.

## Todo
- MinkowskiEngine v0.5
- attribute compression

## Bug Report
- Error on GPU when the point cloud has too many points(>300000). (no error on CPU)

## Authors
These files are provided by Nanjing University  [Vision Lab](https://vision.nju.edu.cn/). And thanks for the help from Hangzhou Normal University. Please contact us (wangjq@smail.nju.edu.cn) if you have any questions. 