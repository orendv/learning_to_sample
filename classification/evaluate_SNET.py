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
parser.add_argument('--classifier_model', default='pointnet_cls', help='Classifier model name [pointnet_cls/pointnet_cls_basic] [default:pointnet_cls]')
parser.add_argument('--sampler_model', default='snet_model', help='Sampler model name: [default: snet_model]')
parser.add_argument('--sampler_model_path', default='log/SNET64/model.ckpt', help='Path to model.ckpt file of S-NET')
parser.add_argument('--num_in_points', type=int, default=1024, help='Number of input Points [default: 1024]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during evaluation [default: 32]')
parser.add_argument('--bottleneck_size', type=int, default=128, help='bottleneck size [default: 128]')
parser.add_argument('--match_output', type=int, default=1, help='Matching flag: 1 - match, 0 - do not match [default:1]')
parser.add_argument('--save_points', type=int, default=0, help='Output points saving flag: 1 - save, 0 - do not save [default:0]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [default:dump]')
parser.add_argument('--num_out_points', type=int, default=64, help='Number of output points [2,4,...,1024] [default: 64]')
FLAGS = parser.parse_args()

GPU_INDEX = FLAGS.gpu
CLASSIFIER_MODEL = importlib.import_module(FLAGS.classifier_model)  # import network module
SAMPLER_MODEL = importlib.import_module(FLAGS.sampler_model)  # import network module
SAMPLER_MODEL_PATH = FLAGS.sampler_model_path
NUM_IN_POINTS = FLAGS.num_in_points
BATCH_SIZE = FLAGS.batch_size
BOTTLENECK_SIZE = FLAGS.bottleneck_size
MATCH_OUTPUT = FLAGS.match_output
SAVE_POINTS = FLAGS.save_points
DUMP_DIR = FLAGS.dump_dir
NUM_OUT_POINTS = FLAGS.num_out_points

model_path, model_file_name = os.path.split(SAMPLER_MODEL_PATH)
OUT_DATA_PATH = model_path

if not os.path.exists(DUMP_DIR): os.makedirs(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')
data_dtype = 'float32'
label_dtype = 'uint8'

NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
               open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))]

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def evaluate():
    with tf.device('/gpu:' + str(GPU_INDEX)):
        pointclouds_pl, labels_pl = CLASSIFIER_MODEL.placeholder_inputs(BATCH_SIZE, NUM_IN_POINTS)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        with tf.variable_scope('sampler'):
            generated_points = SAMPLER_MODEL.get_model(pointclouds_pl, is_training_pl, NUM_OUT_POINTS, BOTTLENECK_SIZE)

        idx = SAMPLER_MODEL.get_nn_indices(pointclouds_pl, generated_points)

        outcloud = generated_points
        pred, end_points = CLASSIFIER_MODEL.get_model(outcloud, is_training_pl)

        loss = CLASSIFIER_MODEL.get_loss(pred, labels_pl, end_points)

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
           'pred': pred,
           'loss': loss,
           'generated_points': generated_points,
           'idx': idx,
           'outcloud': outcloud
           }

    eval_one_epoch(sess, ops)


def eval_one_epoch(sess, ops):
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    num_unique_idx = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')
    for fn in range(len(TEST_FILES)):
        log_string('---- file number ' + str(fn + 1) + ' out of ' + str(len(TEST_FILES)) + ' files ----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:, 0:NUM_IN_POINTS, :]

        current_label_orig = current_label
        current_label = np.squeeze(current_label)
        print(current_data.shape)
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        print(file_size)

        out_data_generated = np.zeros((current_data.shape[0], NUM_OUT_POINTS, current_data.shape[2]))
        out_data_sampled = np.zeros((current_data.shape[0], NUM_OUT_POINTS, current_data.shape[2]))

        for batch_idx in range(num_batches):
            print str(batch_idx) + '/' + str(num_batches - 1)

            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE
            cur_batch_size = end_idx - start_idx

            # Aggregating BEG
            batch_loss_sum = 0  # sum of losses for the batch
            batch_pred_sum = np.zeros((cur_batch_size, NUM_CLASSES))  # score for classes
            batch_pred_classes = np.zeros((cur_batch_size, NUM_CLASSES))  # 0/1 for classes
            rotated_data = current_data[start_idx:end_idx, :, :]
            feed_dict = {ops['pointclouds_pl']: rotated_data, ops['labels_pl']: current_label[start_idx:end_idx], ops['is_training_pl']: is_training}
            generated_points, idx = sess.run([ops['generated_points'], ops['idx']], feed_dict=feed_dict)

            if MATCH_OUTPUT:
                outcloud = SAMPLER_MODEL.nn_matching(rotated_data, idx, NUM_OUT_POINTS)

            else:
                outcloud = generated_points

            for ii in range(0, BATCH_SIZE):
                num_unique_idx += np.size(np.unique(idx[ii]))
            feed_dict = {ops['pointclouds_pl']: rotated_data, ops['outcloud']: outcloud, ops['labels_pl']: current_label[start_idx:end_idx], ops['is_training_pl']: is_training}
            loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)

            out_data_generated[start_idx:end_idx, :, :] = generated_points
            out_data_sampled[start_idx:end_idx, :, :] = outcloud

            batch_pred_sum += pred_val
            batch_pred_val = np.argmax(pred_val, 1)
            for el_idx in range(cur_batch_size):
                batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
            batch_loss_sum += loss_val * cur_batch_size

            pred_val = np.argmax(batch_pred_sum, 1)
            # Aggregating END

            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += cur_batch_size
            loss_sum += batch_loss_sum

            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i - start_idx] == l)
                fout.write('%d, %d\n' % (pred_val[i - start_idx], l))

        if SAVE_POINTS:
            if not os.path.exists(OUT_DATA_PATH + '/generated/'):
                os.makedirs(OUT_DATA_PATH + '/generated/')
            if not os.path.exists(OUT_DATA_PATH + '/sampled/'):
                os.makedirs(OUT_DATA_PATH + '/sampled/')
            file_name = os.path.split(TEST_FILES[fn])
            data_prep_util.save_h5(OUT_DATA_PATH + '/generated/' + file_name[1], out_data_generated, current_label_orig, data_dtype, label_dtype)
            data_prep_util.save_h5(OUT_DATA_PATH + '/sampled/' + file_name[1], out_data_sampled, current_label_orig, data_dtype, abel_dtype)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))
    log_string('total_seen: %f' % (total_seen))

    class_accuracies = np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)
    for i, name in enumerate(SHAPE_NAMES):
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))


if __name__ == '__main__':
    evaluate()
    LOG_FOUT.close()
