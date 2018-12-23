### Installation

Install <a href="https://www.tensorflow.org/get_started/os_setup" target="_blank">TensorFlow</a> and <a href="http://tflearn.org/installation" target="_blank">TFLearn</a>. 

The code has been tested with Python 2.7.12, TensorFlow 1.2.1, TFLearn 0.3.2, CUDA 8.0 and cuDNN 5.1.1 on Ubuntu 16.04.

In order to download the dataset, wget package is required. To install wget:
```
sudo apt-get update
sudo apt-get install wget
```

Compile the structural losses, implemented by [Fan et al.](https://github.com/fanhqme/PointSetGeneration):
```
cd reconstruction/external/structural_losses/
```

If needed, use a text editor and modify the first three lines of the `makefile` to point to your `nvcc`, `cuda` and `tensorflow` library. Then use:
```
make
```

Compile the Farthest Point Sampling operation, implemented by [Qi et al.](https://github.com/charlesq34/pointnet2):
```
cd reconstruction/external/sampling/
```

If needed, use a text editor and modify the `tf_sampling_compile.sh` to point to your `nvcc`, `cuda` and `tensorflow` library. In addition, if you are using a TensorFlow version >= 1.4, add Tensorflow include and library paths to the file:

    TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
    TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())') 

Then, add the flags `-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework` to the `g++` commands. Then use:

    sh tf_sampling_compile.sh

### Data Set
Download point clouds of <a href="https://www.shapenet.org/" target="_blank">ShapeNetCore</a> models in ply files (provided by <a href="https://github.com/optas/latent_3d_points" target="_blank">Achlioptas et al.</a>): 
```
cd reconstruction/
sh download_data.sh
```

The point clouds will be downloaded (1.4GB) to `reconstruction/data/shape_net_core_uniform_samples_2048`. Each point cloud contains 2048 points, uniformly sampled from a shape surface.

### Usage
For quick start please use:

    sh runner_s_net.sh
 
or:

    sh runner_progressive_net.sh

#### Autoencoder

To train an Autoencoder model:

    python autoencoder/train_ae.py --train_folder autoencoder

#### S-NET
To train S-NET using an existing Autoencoder model as the task network (provided in ae_folder flag):

    python sampler/train_s_net.py --ae_folder autoencoder --n_sample_points 64 --train_folder s_net_64

To evaluate reconstruction with S-NET's sampled points (with sample size 64 in this example):

    python sampler/evaluate_s_net.py --train_folder s_net_64

#### ProgressiveNet
To train ProgressiveNet, using an existing Autoencoder model as the task network (provided in ae_folder flag):

    python sampler/train_progressive_net.py --ae_folder autoencoder --train_folder progressive_net

To evaluate reconstruction with ProgressiveNet's sampled points (with sample size 64 in this example):

    python sampler/evaluate_progressive_net.py --n_sample_points 64 --train_folder progressive_net

### Acknowledgment
Our code builds upon the code provided by <a href="https://github.com/optas/latent_3d_points" target="_blank">Achlioptas et al.</a> We would like to thank the authors for sharing their code.
