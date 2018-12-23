import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import data_prep_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--sampler_model', default='snet_model', help='Sampler model name: [default: snet_model]')
parser.add_argument('--sampler_model_path', default='log/ProgressiveNet/model.ckpt')
parser.add_argument('--num_in_points', type=int, default=1024, help='Number of input Points [default: 1024]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during inference [default: 32]')
parser.add_argument('--bottleneck_size', type=int, default=128, help='bottleneck size [default: 128]')
parser.add_argument('--num_generated_points', type=int, default=1024, help='Number of generated Points [default: 1024]')
parser.add_argument('--num_out_points', type=int, default=1024, help='Number of output Points [default: 1024]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--also_infer_train_files', type=int, default=0, help='Flag for including training files in ProgressiveNet inference [default: 0]')
FLAGS = parser.parse_args()

GPU_INDEX = FLAGS.gpu
SAMPLER_MODEL = importlib.import_module(FLAGS.sampler_model)  # import network module
SAMPLER_MODEL_PATH = FLAGS.sampler_model_path
NUM_IN_POINTS = FLAGS.num_in_points
BATCH_SIZE = FLAGS.batch_size
BOTTLENECK_SIZE = FLAGS.bottleneck_size
NUM_GENERATED_POINTS = FLAGS.num_generated_points
NUM_OUT_POINTS = FLAGS.num_out_points
DUMP_DIR = FLAGS.dump_dir
ALSO_INFER_TRAIN_FILES = FLAGS.also_infer_train_files

model_path,model_file_name = os.path.split(SAMPLER_MODEL_PATH)
OUT_DATA_PATH = model_path

if not os.path.exists(DUMP_DIR): os.makedirs(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')
data_dtype = 'float32'
label_dtype = 'uint8'
NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))]
HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

if ALSO_INFER_TRAIN_FILES:
    INFER_FILES = TEST_FILES + TRAIN_FILES
else:
    INFER_FILES = TEST_FILES

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def evaluate():
    with tf.device('/gpu:' + str(GPU_INDEX)):
        pointclouds_pl, labels_pl = SAMPLER_MODEL.placeholder_inputs(BATCH_SIZE, NUM_IN_POINTS)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        with tf.variable_scope('sampler'):
            generated_points = SAMPLER_MODEL.get_model(pointclouds_pl, is_training_pl, NUM_GENERATED_POINTS, BOTTLENECK_SIZE)

        idx = SAMPLER_MODEL.get_nn_indices(pointclouds_pl, generated_points)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, SAMPLER_MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'generated_points': generated_points,
           'idx': idx
           }

    eval_one_epoch(sess, ops)


def eval_one_epoch(sess, ops):
    is_training = False
    total_seen = 0
    for fn in range(len(INFER_FILES)):
        log_string('---- file number ' + str(fn+1) + ' out of ' + str(len(INFER_FILES)) + ' files ----')
        current_data, current_label = provider.loadDataFile(INFER_FILES[fn])
        current_data = current_data[:, 0:NUM_IN_POINTS, :]
        out_data_generated = np.zeros(
            (current_data.shape[0], NUM_GENERATED_POINTS, current_data.shape[2]))

        out_data_sampled_fw_plus_simple_continued_fps = np.zeros(
            (current_data.shape[0], NUM_OUT_POINTS, current_data.shape[2]))

        current_label_orig = current_label
        current_label = np.squeeze(current_label)
        print(current_data.shape)

        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        print(file_size)

        for batch_idx in range(num_batches):
            print str(batch_idx) + '/' + str(num_batches-1)
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE

            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx], ops['labels_pl']: current_label[start_idx:end_idx], ops['is_training_pl']: is_training}
            generated_points, idx = sess.run([ops['generated_points'], ops['idx']], feed_dict=feed_dict)

            out_data_generated[start_idx:end_idx, :, :] = generated_points

            outcloud_fw_plus_simple_continued_fps = SAMPLER_MODEL.nn_matching(current_data[start_idx:end_idx], idx, NUM_OUT_POINTS)
            out_data_sampled_fw_plus_simple_continued_fps[start_idx:end_idx, :, :] = outcloud_fw_plus_simple_continued_fps[:, 0:NUM_OUT_POINTS, :]

            total_seen += BATCH_SIZE

        file_name = os.path.split(INFER_FILES[fn])
        if not os.path.exists(OUT_DATA_PATH + '/generated/'): os.makedirs(OUT_DATA_PATH + '/generated/')
        data_prep_util.save_h5(OUT_DATA_PATH + '/generated/' + file_name[1], out_data_generated, current_label_orig, data_dtype, label_dtype)

        if not os.path.exists(OUT_DATA_PATH + '/sampled/'): os.makedirs(OUT_DATA_PATH + '/sampled/')
        data_prep_util.save_h5(OUT_DATA_PATH + '/sampled/' + file_name[1], out_data_sampled_fw_plus_simple_continued_fps, current_label_orig, data_dtype, label_dtype)


if __name__ == '__main__':
    evaluate()
    LOG_FOUT.close()
