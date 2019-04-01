# Learning to Sample
Created by Oren Dovrat, Itai Lang and Shai Avidan from Tel-Aviv University.

![teaser](https://github.com/orendv/learning_to_sample/blob/master/doc/teaser2.png)

## Introduction
We propose a learned sampling approach for point clouds. Please see our [arXiv tech report](https://arxiv.org/abs/1812.01659).

Processing large point clouds is a challenging task. Therefore, the data is often sampled to a size that can be processed more easily. The question is how to sample the data? A popular sampling technique is Farthest Point Sampling (FPS). However, FPS is agnostic to a downstream application (classification, retrieval, etc.). The underlying assumption seems to be that minimizing the farthest point distance, as done by FPS, is a good proxy to other objective functions. 
We show that it is better to learn how to sample. To do that, we propose a deep network to simplify 3D point clouds. The network, termed S-NET, takes a point cloud and produces a smaller point cloud that is optimized for a particular task. The simplified point cloud is not guaranteed to be a subset of the original point cloud. Therefore, we match it to a subset of the original points in a post-processing step. We contrast our approach with FPS by experimenting on two standard data sets and show significantly better results for a variety of applications.

## Citation
If you find our work useful in your research, please consider citing:

	@article{dovrat2018learning_to_sample,
	  title={Learning to Sample},
	  author={Dovrat, Oren and Lang, Itai and Avidan, Shai},
	  journal={arXiv preprint arXiv:1812.01659},
	  year={2018}
	}

## Installation and usage
This project contains two sub-directories, each is a stand-alone project with it's own instructions.
Please see `classification/README.md` and `reconstruction/README.md`.

## License
This project is licensed under the terms of the MIT license (see `LICENSE` for details).
