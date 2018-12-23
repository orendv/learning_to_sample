### Installation

Install <a href="https://www.tensorflow.org/get_started/os_setup" target="_blank">TensorFlow</a>. You may also need to install h5py. The code has been tested with Python 2.7.12, TensorFlow 1.2.1, CUDA 8.0 and cuDNN 5.1.1 on Ubuntu 16.04.

To install h5py for Python:
```bash
sudo apt-get install libhdf5-dev
sudo pip install h5py
```

Automatic dataset download requiare wget. 
To install wget:
```bash
sudo apt-get update
sudo apt-get install wget
```

Compile the structural losses (only needed for the matching step in inference time), implemented by [Fan et al.](https://github.com/fanhqme/PointSetGeneration):
```
cd classification/structural_losses/
```

If needed, use a text editor and modify the first three lines of the `makefile` to point to your `nvcc`, `cuda` and `tensorflow` library. Then use:
```
make
```

### Usage
For quick start please use:

    sh runner_SNET.sh 
or: 

    sh runner_progressiveNet.sh
    
#### PointNet classifier

To train a PointNet model to classify point clouds:

    python train_classifier.py --log_dir log/baseline/PointNet1024
    
To train the vanilla version of PointNet:

    python train_classifier.py --model pointnet_cls_basic --log_dir log/baseline/PointNetVanilla1024

Point clouds of <a href="http://modelnet.cs.princeton.edu/" target="_blank">ModelNet40</a> models in HDF5 files (provided by <a href="https://github.com/charlesq34/pointnet" target="_blank">Qi et al.</a>) will be automatically downloaded (416MB) to the data folder. Each point cloud contains 2048 points uniformly sampled from a shape surface. Each cloud is zero-mean and normalized into an unit sphere. There are also text files in `data/modelnet40_ply_hdf5_2048` specifying the ids of shapes in h5 files.

#### S-NET
To train S-NET (with sample size 64 in this example), using an existing PointNet classifier as the task network (provided in classifier_model_path):

    python train_SNET.py --classifier_model_path log/baseline/PointNet1024/model.ckpt --num_out_points 64 --log_dir log/SNET64

To infer S-NET and evaluate the classifier over S-NET's sampled points:

    python evaluate_SNET.py --sampler_model_path log/SNET64/model.ckpt --dump_dir log/SNET64/eval

#### ProgressiveNet
To train ProgressiveNet, using an existing classifier (PointNet vanilla in this example) as the task network:

    python train_progressiveNet.py --classifier_model pointnet_cls_basic --classifier_model_path log/baseline/PointNetVanilla1024/model.ckpt --log_dir log/ProgressiveNet

Evlautaion of ProgressiveNet is a two-step process. 
First infer ProgressiveNet and save the ordered point clouds to .h5 files:

    python infer_progressiveNet.py --sampler_model_path log/ProgressiveNet/model.ckpt
    
Second, evaluate the PointNet (vanilla) classifier using ProgressiveNet's sampled points:

    python evaluate_from_files.py --classifier_model pointnet_cls_basic --classifier_model_path log/baseline/PointNetVanilla1024/model.ckpt --data_path log/ProgressiveNet/sampled --dump_dir log/ProgressiveNet/eval

### Acknowledgment
Our code builds upon the code provided by <a href="https://github.com/charlesq34/pointnet" target="_blank">Qi et al.</a> We would like to thank the authors for sharing their code.
