### Installation

Install <a href="https://www.tensorflow.org/get_started/os_setup" target="_blank">TensorFlow</a>. The code has been tested with Python 2.7.12, TensorFlow 1.13.2, CUDA 10.0 and cuDNN 7.6.2 on Ubuntu 16.04.

Install h5py for Python:
```bash
sudo apt-get install libhdf5-dev
sudo pip install h5py
```

Automatic dataset download requires wget. 
To install wget:
```bash
sudo apt-get update
sudo apt-get install wget
```

Compile TensorFlow ops (only needed for the matching step at inference time): structural losses, implemented by [Fan et al.](https://github.com/fanhqme/PointSetGeneration) The ops are located under `classification` at `structural losses` folder. If needed, use a text editor and modify the corresponding `sh` file of each op to point to your `nvcc` path. Then, use:
```bash
cd classification/
sh compile_ops.sh
```

An `o` and `so` files should be created for each op.

### Usage
For quick start please use:
```bash
sh runner_SNET.sh 
```
or: 
```bash
sh runner_progressiveNet.sh
```

These scripts train a classifier model with complete point clouds, use it to train a sampler (S-Net or ProgressiveNet), and then evaluate the sampler by running its sampled points through the classifier model. In the following sections, we explain how to run each part of this pipeline separately.
    
#### PointNet classifier
To train a PointNet model to classify point clouds:
```bash
python train_classifier.py --log_dir log/baseline/PointNet1024
```
    
To train the vanilla version of PointNet:
```bash
python train_classifier.py --model pointnet_cls_basic --log_dir log/baseline/PointNetVanilla1024
```

Point clouds of <a href="http://modelnet.cs.princeton.edu/" target="_blank">ModelNet40</a> models in HDF5 files (provided by <a href="https://github.com/charlesq34/pointnet" target="_blank">Qi et al.</a>) will be automatically downloaded (416MB) to the data folder. Each point cloud contains 2048 points uniformly sampled from a shape surface. Each cloud is zero-mean and normalized into an unit sphere. There are also text files in `data/modelnet40_ply_hdf5_2048` specifying the ids of shapes in h5 files.

#### S-NET
To train S-NET (with sample size 64 in this example), using an existing PointNet classifier as the task network (provided in classifier_model_path):
```bash
python train_SNET.py --classifier_model_path log/baseline/PointNet1024/model.ckpt --num_out_points 64 --log_dir log/SNET64
```

To infer S-NET and evaluate the classifier over S-NET's sampled points:
```bash
python evaluate_SNET.py --sampler_model_path log/SNET64/model.ckpt --dump_dir log/SNET64/eval
```

#### ProgressiveNet
To train ProgressiveNet, using an existing classifier (PointNet vanilla in this example) as the task network:
```bash
python train_progressiveNet.py --classifier_model pointnet_cls_basic --classifier_model_path log/baseline/PointNetVanilla1024/model.ckpt --log_dir log/ProgressiveNet
```

Evaluation of ProgressiveNet is a two-step process. 
First infer ProgressiveNet and save the ordered point clouds to .h5 files:
```bash
python infer_progressiveNet.py --sampler_model_path log/ProgressiveNet/model.ckpt
```

Second, evaluate the PointNet (vanilla) classifier using ProgressiveNet's sampled points:
```bash
python evaluate_from_files.py --classifier_model pointnet_cls_basic --classifier_model_path log/baseline/PointNetVanilla1024/model.ckpt --data_path log/ProgressiveNet/sampled --dump_dir log/ProgressiveNet/eval
```

#### Retrieval
To run a retrieval experiment, please use the following script:
```bash
sh runner_retrieval.sh 
```

In this script, data for retrieval is produced during the inference phase of S-NET. Then, this data is analyzed and corresponding results are saved. 

### Acknowledgment
Our code builds upon the code provided by <a href="https://github.com/charlesq34/pointnet" target="_blank">Qi et al.</a> We would like to thank the authors for sharing their code.
