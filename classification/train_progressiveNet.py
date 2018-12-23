import argparse
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--classifier_model', default='pointnet_cls_basic', help='Classifier model name [pointnet_cls/pointnet_cls_basic] [default:pointnet_cls_basic]')
parser.add_argument('--classifier_model_path', default='log/PointNetVanilla1024/model.ckpt', help='Path to model.ckpt file of a pre-trained classifier')
parser.add_argument('--sampler_model', default='snet_model', help='Sampler model name: [default: snet_model]')
parser.add_argument('--num_in_points', type=int, default=1024, help='Number of input Points [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=500, help='Epoch to run [default: 500]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate [default: 0.01]')
parser.add_argument('--decay_step', type=int, default=600000, help='Decay step for lr decay [default: 600000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--bottleneck_size', type=int, default=128, help='bottleneck size [default: 128]')
parser.add_argument('--alpha', type=int, default=30, help='Sampling regularization loss weight [default: 30]')
parser.add_argument('--gamma', type=float, default=0.5, help='Lb constant regularization loss weight [default: 1 for S-NET, 0.5 for ProgressiveNet]')
parser.add_argument('--delta', type=float, default=1/30, help='Lb linear regularization loss weight [default: 0 for S-NET, 1/30 for ProgressiveNet]')
parser.add_argument('--min_num_out_point', type=int, default=2, help='Point Number [2/4/8...] [default: 2]')
parser.add_argument('--max_num_out_points', type=int, default=1024, help='Point Number [128/256/512/1024] [default: 1024]')
parser.add_argument('--log_dir', default='log/ProgressiveNet', help='Log dir [default: log/ProgressiveNet]')
FLAGS = parser.parse_args()

GPU_INDEX = FLAGS.gpu
CLASSIFIER_MODEL = importlib.import_module(FLAGS.classifier_model)  # import network module
CLASSIFIER_MODEL_PATH = FLAGS.classifier_model_path
SAMPLER_MODEL = importlib.import_module(FLAGS.sampler_model)  # import network module
NUM_IN_POINTS = FLAGS.num_in_points
MAX_EPOCH = FLAGS.max_epoch
BATCH_SIZE = FLAGS.batch_size
BASE_LEARNING_RATE = FLAGS.learning_rate
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
BOTTLENECK_SIZE = FLAGS.bottleneck_size
ALPHA = FLAGS.alpha
GAMMA = FLAGS.gamma
DELTA = FLAGS.delta
MIN_NUM_OUT_POINTS = FLAGS.min_num_out_point
MAX_NUM_OUT_POINTS = FLAGS.max_num_out_points
LOG_DIR = FLAGS.log_dir

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
CLASSIFIER_MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.classifier_model + '.py')
os.system('cp %s %s' % (CLASSIFIER_MODEL_FILE, LOG_DIR))  # bkp of model def
SAMPLER_MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.sampler_model + '.py')
os.system('cp %s %s' % (SAMPLER_MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp train_progressiveNet.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

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


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    tf.reset_default_graph()
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl = CLASSIFIER_MODEL.placeholder_inputs(BATCH_SIZE, NUM_IN_POINTS)

            is_training_classifier_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_classifier_pl)

            is_training_sampler_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_sampler_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            with tf.variable_scope('sampler'):
                generated_points  = SAMPLER_MODEL.get_model(pointclouds_pl, is_training_sampler_pl, MAX_NUM_OUT_POINTS, BOTTLENECK_SIZE, bn_decay=bn_decay)

            with tf.variable_scope('classifier'):
                pc_size = MIN_NUM_OUT_POINTS
                generated_points_slice = tf.slice(generated_points, [0, 0, 0], [BATCH_SIZE, pc_size, 3])
                pred, end_points = CLASSIFIER_MODEL.get_model(generated_points_slice, is_training_classifier_pl, bn_decay=bn_decay)
                loss_classifier = CLASSIFIER_MODEL.get_loss(pred, labels_pl, end_points)
                loss_sampling = SAMPLER_MODEL.get_sampling_loss(pointclouds_pl, generated_points_slice, pc_size, GAMMA, DELTA)

                correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
                accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
                tf.summary.scalar('accuracy' + str(pc_size), accuracy)

            b = 2*MIN_NUM_OUT_POINTS
            pc_sizes = np.array([], dtype=int)
            while b <= MAX_NUM_OUT_POINTS:
                pc_sizes = np.append(pc_sizes, b)
                b = b*2

            for pc_size in pc_sizes:
                generated_points_slice = tf.slice(generated_points, [0, 0, 0], [BATCH_SIZE, pc_size, 3])
                with tf.variable_scope('classifier' + str(pc_size)):
                    pred, end_points = CLASSIFIER_MODEL.get_model(generated_points_slice, is_training_classifier_pl, bn_decay=bn_decay)
                    loss_curr_classifier = CLASSIFIER_MODEL.get_loss(pred, labels_pl, end_points)
                    loss_classifier = tf.add(loss_classifier, loss_curr_classifier)

                    loss_curr_similarity = SAMPLER_MODEL.get_sampling_loss(pointclouds_pl, generated_points_slice, pc_size, GAMMA, DELTA)

                    loss_sampling = tf.add(loss_sampling, loss_curr_similarity)

                    correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
                    accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
                    tf.summary.scalar('accuracy' + str(pc_size) , accuracy)

            loss = loss_classifier + ALPHA * loss_sampling

            tf.summary.scalar('loss', loss)
            tf.summary.scalar('loss_classifier', loss_classifier)
            tf.summary.scalar('loss_sampling', loss_sampling)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_vars = tf.trainable_variables()
            sampler_params = [v for v in train_vars if v.name.startswith('sampler')]
            train_op = optimizer.minimize(loss, global_step=batch, var_list=sampler_params)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writer
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        # sess.run(init)
        sess.run(init, {is_training_sampler_pl: True, is_training_classifier_pl: False})

        # Restore variables from disk.
        all_variables = tf.get_collection_ref(tf.GraphKeys.GLOBAL_VARIABLES)
        sess.run(tf.variables_initializer(all_variables))

        no_scope_param = list()
        #no_scope_param.append(tf.get_default_graph().get_tensor_by_name("Variable:0"))
        no_scope_param.append(tf.get_default_graph().get_tensor_by_name("beta1_power:0"))
        no_scope_param.append(tf.get_default_graph().get_tensor_by_name("beta2_power:0"))
        classifier_loader = tf.train.Saver(var_list=no_scope_param)
        classifier_loader.restore(sess, CLASSIFIER_MODEL_PATH)

        restore_into_scope(CLASSIFIER_MODEL_PATH, 'classifier', sess)
        for pc_size in pc_sizes:
            restore_into_scope(CLASSIFIER_MODEL_PATH, 'classifier' + str(pc_size), sess)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_sampler_pl': is_training_sampler_pl,
               'is_training_classifier_pl': is_training_classifier_pl,
               'pred': pred,
               'loss': loss,
               'loss_classifier': loss_classifier,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'generated_points': generated_points
               }

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops)

            # Save the variables to disk.
            if epoch % 10 == 0 or epoch == MAX_EPOCH - 1:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training_sampler = True
    is_training_classifier = False

    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)

    for fn in range(len(TRAIN_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:, 0:NUM_IN_POINTS, :]
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
        current_label = np.squeeze(current_label)
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE

        total_correct = 0
        total_seen = 0
        loss_sum = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE

            # Augment batched point clouds by rotation and jittering
            rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)

            feed_dict = {ops['pointclouds_pl']: jittered_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_sampler_pl']: is_training_sampler,
                         ops['is_training_classifier_pl']: is_training_classifier}

            if is_training_classifier:
                summary, step, _, loss_val, pred_val = sess.run(
                    [ops['merged'], ops['step'], ops['train_op'], ops['loss_classifier'], ops['pred']],
                    feed_dict=feed_dict)
            else:
                summary, step, _, loss_val, pred_val = sess.run(
                    [ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                    feed_dict=feed_dict)

            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_val

        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('accuracy: %f' % (total_correct / float(total_seen)))


def eval_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training_sampler = False
    is_training_classifier = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    for fn in range(len(TEST_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:, 0:NUM_IN_POINTS, :]
        current_label = np.squeeze(current_label)

        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE

            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_sampler_pl']: is_training_sampler,
                         ops['is_training_classifier_pl']: is_training_classifier}
            summary, step, loss_val, pred_val\
                = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']
                            ], feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += (loss_val * BATCH_SIZE)
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i - start_idx] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (
    np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))
    log_string('total_seen: %f' % (total_seen))


def restore_into_scope(model_path, scope_name, sess):
    # restored_vars = get_tensors_in_checkpoint_file(file_name=MODEL_PATH)
    # tensors_to_load = build_tensors_in_checkpoint_file(restored_vars, scope=scope_name)

    global_vars = tf.global_variables()
    tensors_to_load = [v for v in global_vars if v.name.startswith(scope_name + '/')]

    load_dict = {}
    for j in range(0, np.size(tensors_to_load)):
        tensor_name = tensors_to_load[j].name
        tensor_name = tensor_name[0:-2] # remove ':0'
        tensor_name = tensor_name.replace(scope_name + '/', '') #remove scope
        load_dict.update({tensor_name: tensors_to_load[j]})
    loader = tf.train.Saver(var_list=load_dict)
    loader.restore(sess, model_path)
    log_string("Model restored from: {0} into scope: {1}.".format(model_path, scope_name))


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
