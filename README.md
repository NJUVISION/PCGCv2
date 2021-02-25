# Multiscale Point Cloud Geometry Compression

​	We apply an **end-to-end learning framework** to compress the 3D  point cloud geometry (PCG) efficiently. Leveraging the sparsity nature of point cloud, we introduce the **multiscale structure** to represent native PCG compactly, offering the **hierarchical reconstruction** capability via progressive learnt re-sampling. Under this framework, we devise the **sparse convolution-based autoencoder** for feature analysis and aggregation. At the bottleneck layer, geometric occupancy information is losslessly encoded with a very small percentage of bits consumption, and corresponding feature attributes are lossy compressed. 

## News

- 2021.1.1 Our paper has been accepted by **DCC2021**! [[paper](https://arxiv.org/abs/2011.03799)]  [[presentation](https://sigport.org/documents/multiscale-point-cloud-geometry-compression)]
- 2021.2.25 We have updated MinkowskiEngine to v0.5. The bug on GPU is fixed. And the encoding and decoding runtime is reduced.

## Framework

<p align="center">
  <img src="figs/framework.png?raw=true" alt="introduction" width="800"/> 

  **Multiscale PCG Compression:**  (a) "Conv c*n^3" denotes the sparse convolution with 'c' output channels and n^3 kernel size, "Q" stands for Quantization, "AE" and "AD" are Arithmetic Encoder and Decoder respectively. "IRN" is Inception-Residual Network used for efficient feature aggregation. (a) network structure of IRN unit, (c) adaptive contexts conditioned on autoregressive priors.
  <img src="figs/reconstruct.png?raw=true" alt="introduction" width="800"/> 

​	**Binary classification based hierarchical reconstruction**: The top part shows the encoding process: **(a), (b), (c), (d)** are exemplified from a size of 32^3 to  4^3, by halving each geometric dimension scale step-by-step; The bottom part illustrates corresponding  hierarchical reconstructions, geometric models are upscaled and classified gradually from the rightmost to the leftmost position: **(e), (g), (i)** are convolutionally upscaled from lower scales with size of  8^3, 16^3 and 32^3. Different colors are used to differentiate the **probabilities of voxel-being-occupied** (i.e., the greener, the closer  to 1, and the bluer, the closer to 0); and **(f), (h), (j)** are the reconstructions after classification with green blocks for true classified voxels,  blue for false positive, and red for false negative voxels.
</p>



## Requirments

- python3.7

- cuda10.2 

- pytorch1.7

- MinkowskiEngine 0.5 

  We recommend you to follow the [MinkowskiEngine installation instruction](https://github.com/stanfordvl/MinkowskiEngine) to setup the environment for the sparse convolution. 

- tensorflow1.13 (for Arithmetic Encoder)

- Pretrained Models: https://box.nju.edu.cn/f/46d9206c6565471fb256/
- Results: https://box.nju.edu.cn/f/f2757a55e5e94440b2a7/
- Testdata: https://box.nju.edu.cn/f/e7a4578decf24cfa8e09/
- Training Dataset: http://yun.nju.edu.cn/f/7c81b0e501/

## Usage

### Training
```shell
 python train.py --dataset='training_dataset/' --dataset_8i = 'testdata/8iVFB/'
```

### Testing
```shell
sudo chmod 777 utils/tmc3
python eval.py --filedir='testdata/8iVFB/redandblack_vox10_1550.ply' --ckptdir='ckpts/c8_a2_32000.pth'
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
- 2021.02.25 bug fixed.

## Authors
These files are provided by Nanjing University  [Vision Lab](https://vision.nju.edu.cn/). And thanks for the help from Prof. Dandan Ding from Hangzhou Normal University and Prof. Zhu Li from University of Missouri at Kansas. Please contact us (mazhan@nju.edu.cn and wangjq@smail.nju.edu.cn) if you have any questions.