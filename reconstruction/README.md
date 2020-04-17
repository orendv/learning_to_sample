### Installation

Install <a href="https://www.tensorflow.org/get_started/os_setup" target="_blank">TensorFlow</a> and <a href="http://tflearn.org/installation" target="_blank">TFLearn</a>. 

The code has been tested with Python 2.7.12, TensorFlow 1.13.2, TFLearn 0.3.2, CUDA 10.0 and cuDNN 7.6.2 on Ubuntu 16.04.

In order to download the dataset, wget package is required. To install wget:
```bash
sudo apt-get update
sudo apt-get install wget
```

Compile TensorFlow ops: farthest point sampling, implemented by [Qi et al.](https://github.com/charlesq34/pointnet2); structural losses, implemented by [Fan et al.](https://github.com/fanhqme/PointSetGeneration) The ops are located under `reconstruction/external` at `sampling`, and `structural losses` folders, respectively. If needed, use a text editor and modify the corresponding `sh` file of each op to point to your `nvcc` path. Then, use:   
```bash
cd reconstruction/
sh compile_ops.sh
```

An `o` and `so` files should be created in the corresponding folder of each op. 

### Data Set
Download point clouds of <a href="https://www.shapenet.org/" target="_blank">ShapeNetCore</a> models in ply files (provided by <a href="https://github.com/optas/latent_3d_points" target="_blank">Achlioptas et al.</a>): 
```bash
cd reconstruction/
sh download_data.sh
```

The point clouds will be downloaded (1.4GB) to `reconstruction/data/shape_net_core_uniform_samples_2048`. Each point cloud contains 2048 points, uniformly sampled from a shape surface.

### Usage
For quick start please use:
```bash
sh runner_s_net.sh
```
or:
```bash
sh runner_progressive_net.sh
```

These scripts train and evaluate an Autoencoder model with complete point clouds, use it to train a sampler (S-Net or ProgressiveNet), and then evaluate the sampler by running its sampled points through the Autoencoder model. In the following sections, we explain how to run each part of this pipeline separately.

#### Autoencoder

To train an Autoencoder model, use:
```bash
python autoencoder/train_ae.py --train_folder log/autoencoder
```

To evaluate the Autoencoder model, use:
```bash
python autoencoder/evaluate_ae.py --train_folder log/autoencoder
```

This evaluation script saves the reconstructed point clouds from complete input point clouds of the test set, and the reconstruction error per point cloud (Chamfer distance between the input and reconstruction). The results are saved to the `train_folder`.

To evaluate reconstruction with FPS sampled points (with sample size 64 in this example), use:  
```bash
python autoencoder/evaluate_ae.py --train_folder log/autoencoder --use_fps 1 --n_sample_points 64
```

This evaluation script saves the sampled point clouds, sample indices and reconstructed point clouds of the test set, and the reconstruction error per point cloud (Chamfer distance between the input and reconstruction). It also computes the normalized reconstruction error, as explained in the paper. The results are saved to the `train_folder`.

#### S-NET
To train S-NET using an existing Autoencoder model as the task network (provided in ae_folder flag), use:
```bash
python sampler/train_s_net.py --ae_folder autoencoder --n_sample_points 64 --train_folder log/s_net_64
```

To evaluate reconstruction with S-NET's sampled points (with sample size 64 in this example), use:
```bash
python sampler/evaluate_s_net.py --train_folder log/s_net_64
```

This script operates similarly to the evaluation script for the Autoencoder with FPS sampled points.

#### ProgressiveNet
To train ProgressiveNet, using an existing Autoencoder model as the task network (provided in ae_folder flag), use:
```bash
python sampler/train_progressive_net.py --ae_folder log/autoencoder --n_sample_points 64 --train_folder log/progressive_net
```

To evaluate reconstruction with ProgressiveNet's sampled points (with sample size 64 in this example), use:
```bash
python sampler/evaluate_progressive_net.py --n_sample_points 64 --train_folder log/progressive_net
```

This script operates similarly to the evaluation script for the Autoencoder with FPS sampled points.

#### Visualization
You can visualized point clouds (input, reconstructed, or sampled) by adding the flag `--visualize_results` to the evaluation script of the Autoencoder, S-NET or ProgressiveNet.

### Acknowledgment
Our code builds upon the code provided by <a href="https://github.com/optas/latent_3d_points" target="_blank">Achlioptas et al.</a> We would like to thank the authors for sharing their code.
