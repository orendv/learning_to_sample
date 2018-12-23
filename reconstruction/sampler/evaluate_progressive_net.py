"""
Created on September 10th, 2018

@author: itailang
"""

# import system modules
import os.path as osp
import sys
import argparse
import numpy as np

# add paths
parent_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import modules
from reconstruction.src.autoencoder import Configuration as Conf
from reconstruction.src.sample_net_progressive_point_net_ae import SampleNetProgressivePointNetAutoEncoder

from reconstruction.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_and_split_all_point_clouds_under_folder

from reconstruction.src.tf_utils import reset_tf_graph
from reconstruction.src.general_utils import plot_3d_point_cloud


# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_sample_points', type=int, default=64, help='Number of sample points (for evluation) [default: 64]')
parser.add_argument('--object_class', type=str, default='multi', help='Single class name (for example: chair) or multi [default: multi]')
parser.add_argument('--train_folder', type=str, default='progressive_net', help='Folder for loading data form the training [default: progressive_net]')
flags = parser.parse_args()

print('Evaluate flags:', flags)

# Define basic parameters
project_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
top_in_dir = osp.join(project_dir, 'data', 'shape_net_core_uniform_samples_2048')  # Top-dir of where point-clouds are stored.
top_out_dir = osp.join(project_dir, 'results')                                     # Use to save Neural-Net check-points etc.

if flags.object_class == 'multi':
    class_name = ['chair', 'table', 'car', 'airplane']
    class_name_dir = flags.object_class
else:
    class_name = [str(flags.object_class)]
    class_name_dir = class_name[0]

# Load Point-Clouds
syn_id = snc_category_to_synth_id()[class_name[0]]
class_dir = osp.join(top_in_dir, syn_id)
_, _, pc_data_test = load_and_split_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)

for i in range(1, len(class_name)):
    syn_id = snc_category_to_synth_id()[class_name[i]]
    class_dir = osp.join(top_in_dir, syn_id)
    _, _, pc_data_test_curr = load_and_split_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)
    pc_data_test.merge(pc_data_test_curr)

# Load configuration
train_dir = osp.join(top_out_dir, flags.train_folder)
restore_epoch = 500
conf = Conf.load(osp.join(train_dir, 'configuration'))

conf.pc_size = [flags.n_sample_points]
conf.n_samp = [flags.n_sample_points, 3]

# Reload a saved model
reset_tf_graph()
ae = SampleNetProgressivePointNetAutoEncoder(conf.experiment_name, conf)
ae.restore_model(train_dir, epoch=restore_epoch, verbose=True)

# Input point clouds
complete_pc, _, _ = pc_data_test.next_batch(10)

# Sample points
n_sample_points = conf.n_samp[0]
ordered_pc = ae.sample(complete_pc)[1]  # complete pc ordered by ProgressiveNet
sampled_pc = ordered_pc[:, :n_sample_points, :]  # sample form ProgressiveNet points

sorted_pc = ae.sess.run(ae.x_sorted, feed_dict={ae.x: complete_pc})  # complete pc sorted by FPS
fps_pc = sorted_pc[:, :n_sample_points, :]  # FPS points

# Reconstruct
reconstructed_from_sampled = ae.reconstruct(complete_pc, S=ordered_pc)[0]
reconstructed_from_fps = ae.reconstruct(complete_pc, S=sorted_pc)[0]

# Use any plotting mechanism, such as matplotlib, to visualize the results
i = 0
plot_3d_point_cloud(complete_pc[i][:, 0], complete_pc[i][:, 1], complete_pc[i][:, 2],
                    in_u_sphere=True, title='Complete input point cloud')
plot_3d_point_cloud(sampled_pc[i][:, 0], sampled_pc[i][:, 1], sampled_pc[i][:, 2],
                    in_u_sphere=True, title='ProgressiveNet sampled points')
plot_3d_point_cloud(fps_pc[i][:, 0], fps_pc[i][:, 1], fps_pc[i][:, 2],
                    in_u_sphere=True, title='FPS sampled points')
plot_3d_point_cloud(reconstructed_from_sampled[i][:, 0], reconstructed_from_sampled[i][:, 1], reconstructed_from_sampled[i][:, 2],
                    in_u_sphere=True, title='Reconstruction from ProgressiveNet sampled points')
plot_3d_point_cloud(reconstructed_from_fps[i][:, 0], reconstructed_from_fps[i][:, 1], reconstructed_from_fps[i][:, 2],
                    in_u_sphere=True, title='Reconstruction from FPS sampled points')
