"""
Created on September 5th, 2018

@author: itailang
"""

# import system modules
import os.path as osp
import sys
import argparse

# add paths
parent_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# import modules
from reconstruction.src.samplers import sampler_with_convs_and_symmetry_and_fc
from reconstruction.src.autoencoder import Configuration as Conf
from reconstruction.src.progressive_net_point_net_ae import ProgressiveNetPointNetAutoEncoder

from reconstruction.src.in_out import snc_category_to_synth_id, create_dir, load_and_split_all_point_clouds_under_folder

from reconstruction.src.tf_utils import reset_tf_graph

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--n_sample_points', type=int, default=64, help='Number of sample points (for evluation) [default: 64]')
parser.add_argument('--similarity_reg_weight', type=float, default=0.01, help='Weight of similarity regularization [default: 0.01]')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate [default: 0.0005]')
parser.add_argument('--restore_ae', type=bool, default=True, help='Restore a trained autoencoder model [default: True]')
parser.add_argument('--fixed_ae', type=bool, default=True, help='Fixed autoencoder model [default: True]')
parser.add_argument('--object_class', type=str, default='multi', help='Single class name (for example: chair) or multi [default: multi]')
parser.add_argument('--ae_folder', type=str, default='log/autoencoder', help='Folder for loading a trained autoencoder model [default: log/autoencoder]')
parser.add_argument('--train_folder', type=str, default='log/progressive_net', help='Folder for saving data form the training [default: log/progressive_net]')
flags = parser.parse_args()

print('Train flags:', flags)

# Define basic parameters
project_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
top_in_dir = osp.join(project_dir, 'data', 'shape_net_core_uniform_samples_2048')  # Top-dir of where point-clouds are stored.
top_out_dir = osp.join(project_dir, 'results')                                     # Use to save Neural-Net check-points etc.

if flags.object_class == 'multi':
    class_name = ['chair', 'table', 'car', 'airplane']
else:
    class_name = [str(flags.object_class)]

# Load Point-Clouds
syn_id = snc_category_to_synth_id()[class_name[0]]
class_dir = osp.join(top_in_dir, syn_id)
pc_data_train, pc_data_val, _ = load_and_split_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)

for i in range(1, len(class_name)):
    syn_id = snc_category_to_synth_id()[class_name[i]]
    class_dir = osp.join(top_in_dir, syn_id)
    pc_data_train_curr, pc_data_val_curr, _ = load_and_split_all_point_clouds_under_folder(class_dir, n_threads=8, file_ending='.ply', verbose=True)
    pc_data_train.merge(pc_data_train_curr)
    pc_data_val.merge(pc_data_val_curr)

if flags.object_class == 'multi':
    pc_data_train.shuffle_data(seed=55)
    pc_data_val.shuffle_data(seed=55)

# Load autoencoder configuration
ae_dir = osp.join(top_out_dir, flags.ae_folder)
conf = Conf.load(osp.join(ae_dir, 'configuration'))

# Update autoencoder configuration
conf.ae_dir = ae_dir
conf.ae_name = 'autoencoder'
conf.restore_ae = flags.restore_ae
conf.ae_restore_epoch = 500
conf.fixed_ae = flags.fixed_ae
if conf.fixed_ae:
    conf.encoder_args['b_norm_decay'] = 1.          # for avoiding the update of batch normalization moving_mean and moving_variance parameters
    conf.decoder_args['b_norm_decay'] = 1.          # for avoiding the update of batch normalization moving_mean and moving_variance parameters
    conf.decoder_args['b_norm_decay_finish'] = 1.   # for avoiding the update of batch normalization moving_mean and moving_variance parameters

# sampler configuration
conf.experiment_name = 'sampler'
conf.pc_size = [2**i for i in range(4, 12)]  # Different sample sizes (for training)
conf.n_samp = [flags.n_sample_points, 3]  # Dimensionality of sampled points (for evaluation)
conf.sampler = sampler_with_convs_and_symmetry_and_fc
conf.similarity_reg_weight = flags.similarity_reg_weight
conf.learning_rate = flags.learning_rate

train_dir = create_dir(osp.join(top_out_dir, flags.train_folder))
conf.train_dir = train_dir

conf.save(osp.join(train_dir, 'configuration'))

# Build Sampler and AE Model
reset_tf_graph()
ae = ProgressiveNetPointNetAutoEncoder(conf.experiment_name, conf)

# Train the sampler (save output to train_stats.txt)
buf_size = 1  # Make 'training_stats' file to flush each output line regarding training.
fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
train_stats = ae.train(pc_data_train, conf, log_file=fout, held_out_data=pc_data_val)
fout.close()
