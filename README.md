# Learning to Sample
Created by Oren Dovrat, Itai Lang and Shai Avidan from Tel-Aviv University.

![teaser](https://github.com/orendv/learning_to_sample/blob/master/doc/teaser2.png)

## Introduction
We propose a learned sampling approach for point clouds. Please see our [arXiv tech report](https://arxiv.org/abs/1812.01659).

Processing large point clouds is a challenging task. Therefore, the data is often sampled to a size that can be processed more easily. The question is how to sample the data? A popular sampling technique is Farthest Point Sampling (FPS). However, FPS is agnostic to a downstream application (classification, retrieval, etc.). The underlying assumption seems to be that minimizing the farthest point distance, as done by FPS, is a good proxy to other objective functions. 
We show that it is better to learn how to sample. To do that, we propose a deep network to simplify 3D point clouds. The network, termed S-NET, takes a point cloud and produces a smaller point cloud that is optimized for a particular task. The simplified point cloud is not guaranteed to be a subset of the original point cloud. Therefore, we match it to a subset of the original points in a post-processing step. We contrast our approach with FPS by experimenting on two standard data sets and show significantly better results for a variety of applications.

![poster](https://github.com/orendv/learning_to_sample/blob/master/doc/Learning_to_Sample_poster.PNG)

## Citation
If you find our work useful in your research, please consider citing:

	@InProceedings{dovrat2019learning_to_sample,
	  author = {Dovrat, Oren and Lang, Itai and Avidan, Shai},
	  title = {{Learning to Sample}},
	  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	  pages = {2760--2769},
	  year = {2019}
	}

## Installation and usage
This project contains two sub-directories, each is a stand-alone project with it's own instructions.
Please see `classification/README.md` and `reconstruction/README.md`.

## License
This project is licensed under the terms of the MIT license (see `LICENSE` for details).

## Selected projects that use "Learning to Sample"
* <a href="https://arxiv.org/abs/1912.03663" target="_blank">SampleNet: Differentiable Point Cloud Sampling</a> by Lang *et al*. (CVPR 2020 Oral). This work extends "Learning to Sample" and proposes a novel differentiable relaxation for point cloud sampling.
* <a href="https://www.semanticscholar.org/paper/Multi-Stage-Point-Completion-Network-with-Critical-Zhang-Long/eee0f1cba1dd44b01bb370806359cd64a5a7b50d" target="_blank">Multi-Stage Point Completion Network with Critical Set Supervision</a> by Zhang *et al*. (submitted to CAGD; Special Issue of GMP 2020). This work evaluates our learned sampling as a supervision signal for point cloud completion network.
* <a href="https://arxiv.org/abs/2005.00383" target="_blank">MOPS-Net: A Matrix Optimization-driven Network for Task-Oriented 3D Point Cloud Downsampling</a> by Qian *et al*. (arXiv  preprint). This work suggests an alternative network architecture for learned point cloud sampling. To train their network, the authors use our proposed losses for S-NET and ProgressiveNet. 
