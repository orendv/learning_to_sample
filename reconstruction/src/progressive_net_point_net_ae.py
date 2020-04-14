'''
Created on September 7th, 2018

@author: itailang
'''

import time
import tensorflow as tf
import numpy as np
import os.path as osp

from tflearn.layers.conv import conv_1d
from tflearn.layers.core import fully_connected

from . in_out import create_dir
from . sampler_progressive_autoencoder import SamplerProgressiveAutoEncoder
from . general_utils import apply_augmentations

try:
    from .. external.sampling.tf_sampling import farthest_point_sample, gather_point
except:
    print('Farthest Point Sample cannot be loaded. Please install it first.')

try:    
    from .. external.structural_losses.tf_nndistance import nn_distance
    from .. external.structural_losses.tf_approxmatch import approx_match, match_cost
except:
    print('External Losses (Chamfer-EMD) cannot be loaded. Please install them first.')
    

class ProgressiveNetPointNetAutoEncoder(SamplerProgressiveAutoEncoder):
    '''
    An Auto-Encoder for point-clouds.
    '''

    def __init__(self, sampler_name, configuration, graph=None):
        c = configuration
        self.configuration = c

        SamplerProgressiveAutoEncoder.__init__(self, sampler_name, graph, configuration)

        with tf.variable_scope(sampler_name):
            n_pc_point = self.x.get_shape().as_list()[1]  # number of input points
            n_point_ch = self.x.get_shape().as_list()[2]  # number of channels per point (should be 3)

            idx_fps = farthest_point_sample(n_pc_point, self.x)  # (batch_size, n_pc_point)
            self.x_sorted = gather_point(self.x, idx_fps)  # (batch_size, n_pc_point, 3)

            self._create_match_samples()  # for samples interpolation

            if self.is_denoising:
                self._create_match_cost()

            self.s = c.sampler(self.x, [n_pc_point, n_point_ch])

            _, self.idx, _, _ = nn_distance(self.s, self.x)

        pc_size = c.pc_size
        reuse = [None] + [True] * (len(pc_size) - 1)
        x_reconstr = [None] * len(pc_size)
        loss_ae = [None] * len(pc_size)
        loss_similarity = [None] * len(pc_size)
        nn_dist = [None] * len(pc_size)

        for i in range(len(pc_size)):
            with tf.variable_scope(c.ae_name, reuse=reuse[i]):
                s_slice = self.s[:, :pc_size[i], :]
                self.z = c.encoder(s_slice, **c.encoder_args)
                self.bottleneck_size = int(self.z.get_shape()[1])
                layer = c.decoder(self.z, **c.decoder_args)

                if c.exists_and_is_not_none('close_with_tanh'):
                    layer = tf.nn.tanh(layer)

                x_reconstr[i] = tf.reshape(layer, [-1, self.n_output[0], self.n_output[1]])

            with tf.variable_scope(sampler_name):
                loss_ae[i] = self._get_ae_loss(x_reconstr[i])
                loss_similarity[i], _, _, _, nn_dist[i] = self._get_similarity_loss(self.gt, s_slice, pc_size[i])

        self.x_reconstr = x_reconstr[0]
        self.nn_distance = nn_dist[0]

        with tf.variable_scope(sampler_name):
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=c.saver_max_to_keep)

            self._create_loss(loss_ae, loss_similarity)
            self._setup_optimizer()

            # GPU configuration
            if hasattr(c, 'allow_gpu_growth'):
                growth = c.allow_gpu_growth
            else:
                growth = True

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = growth

            # Summaries
            self.merged_summaries = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(osp.join(configuration.train_dir, 'summaries'), self.graph)

            # Initializing the tensor flow variables
            self.init = tf.global_variables_initializer()

            # Launch the session
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

            if c.restore_ae:
                self.restore_ae_model(c.ae_dir, c.ae_name, c.ae_restore_epoch, verbose=True)

    def _create_match_samples(self):
        match = approx_match(self.s1, self.s2)
        s1_match_idx = tf.cast(tf.argmax(match, axis=2), dtype=tf.int32)
        self.s1_matched = gather_point(self.s1, s1_match_idx)  # self.s1_matched has the shape of self.s2

    def _create_match_cost(self):
        match = approx_match(self.pc1, self.pc2)
        self.match_cost = tf.reduce_mean(match_cost(self.pc1, self.pc2, match))

    def _get_ae_loss(self, x_reconstr):
        c = self.configuration

        # reconstruction loss
        if c.loss == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(x_reconstr, self.gt)
            loss_ae = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
        elif c.loss == 'emd':
            match = approx_match(self.x_reconstr, self.gt)
            loss_ae = tf.reduce_mean(match_cost(x_reconstr, self.gt, match))

        return loss_ae

    def _create_loss(self, loss_ae, loss_similarity):
        c = self.configuration

        self.loss_ae = loss_ae[0]
        self.loss_similarity = loss_similarity[0]

        loss_ae_avg = tf.add_n(loss_ae) / (len(loss_ae)/1.0)
        loss_similarity_avg = tf.add_n(loss_similarity) / (len(loss_similarity)/1.0)

        if c.similarity_reg_weight > 0.0:
            self.loss = loss_ae_avg + c.similarity_reg_weight * loss_similarity_avg
        else:
            self.loss = loss_ae_avg

        reg_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if c.exists_and_is_not_none('w_reg_alpha'):
            w_reg_alpha = c.w_reg_alpha
        else:
            w_reg_alpha = 1.0

        for rl in reg_losses:
            self.loss += (w_reg_alpha * rl)

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('loss_ae', self.loss_ae)
        tf.summary.scalar('loss_similarity', self.loss_similarity)

    def _get_similarity_loss(self, ref_pc, samp_pc, pc_size):
        cost_p1_p2, idx, cost_p2_p1, _ = nn_distance(samp_pc, ref_pc)
        dist = cost_p1_p2
        dist2 = cost_p2_p1
        max_cost = tf.reduce_max(cost_p1_p2, axis=1)
        max_cost = tf.reduce_mean(max_cost)

        nn_dist = tf.reduce_mean(cost_p1_p2, axis=1, keep_dims=True) + tf.reduce_mean(cost_p2_p1, axis=1, keep_dims=True)

        cost_p1_p2 = tf.reduce_mean(cost_p1_p2)
        cost_p2_p1 = tf.reduce_mean(cost_p2_p1)

        w = pc_size / 64.0
        if self.is_denoising:
            loss = cost_p1_p2 + max_cost + 2*w * cost_p2_p1
        else:
            loss = cost_p1_p2 + max_cost + w * cost_p2_p1

        tf.summary.scalar('cost_p1_p2', cost_p1_p2)
        tf.summary.scalar('cost_p2_p1', cost_p2_p1)
        tf.summary.scalar('max_cost', max_cost)

        return loss, dist, idx, dist2, nn_dist

    def _setup_optimizer(self):
        c = self.configuration
        self.lr = c.learning_rate
        if hasattr(c, 'exponential_decay'):
            self.lr = tf.train.exponential_decay(c.learning_rate, self.epoch, c.decay_steps, decay_rate=0.5, staircase=True, name="learning_rate_decay")
            self.lr = tf.maximum(self.lr, 1e-5)
            tf.summary.scalar('learning_rate', self.lr)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

        train_vars = tf.trainable_variables()
        sampler_vars = [v for v in train_vars if v.name.startswith(c.experiment_name)]

        if c.fixed_ae:
            self.train_step = self.optimizer.minimize(self.loss, var_list=sampler_vars)
        else:
            self.train_step = self.optimizer.minimize(self.loss)

    def _single_epoch_train(self, train_data, configuration, only_fw=False, project_points=False, sort_points=True):
        n_examples = train_data.num_examples
        epoch_loss = 0.
        epoch_loss_ae = 0.
        epoch_loss_similarity = 0.
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()

        if only_fw:
            fit = self.reconstruct
        else:
            fit = self.partial_fit

        # Loop over all batches
        for _ in xrange(n_batches):

            if self.is_denoising:
                original_data, _, batch_i = train_data.next_batch(batch_size)
                if batch_i is None:  # In this case the denoising concern only the augmentation.
                    batch_i = original_data
            else:
                original_data = None
                batch_i, _, _ = train_data.next_batch(batch_size)

            batch_i = apply_augmentations(batch_i, configuration)   # This is a new copy of the batch.

            if only_fw and project_points:
                if sort_points:
                    batch_i_sorted = self.sort(batch_i)
                    batch_i = batch_i_sorted

                _, samp_i = self.sample(batch_i)

                _, loss, loss_ae, loss_similarity = fit(batch_i, original_data, S=samp_i)
            else:
                _, loss, loss_ae, loss_similarity = fit(batch_i, original_data)

            # Compute average loss
            epoch_loss += loss
            epoch_loss_ae += loss_ae
            epoch_loss_similarity += loss_similarity
        epoch_loss /= n_batches
        epoch_loss_ae /= n_batches
        epoch_loss_similarity /= n_batches
        duration = time.time() - start_time
        
        if configuration.loss == 'emd':
            epoch_loss_ae /= len(train_data.point_clouds[0])

        epoch_loss = epoch_loss_ae + configuration.similarity_reg_weight * epoch_loss_similarity

        return epoch_loss, epoch_loss_ae, epoch_loss_similarity, duration

    def gradient_of_input_wrt_loss(self, in_points, gt_points=None):
        if gt_points is None:
            gt_points = in_points
        return self.sess.run(tf.gradients(self.loss, self.x), feed_dict={self.x: in_points, self.gt: gt_points})

    def get_sampled_cloud(self, full_pc, gen_pc, idx, dist2):
        batch_size = np.size(full_pc, 0)
        k = np.size(gen_pc, 1)
        out_pc = np.zeros_like(gen_pc)
        for ii in range(0, batch_size):
            nn_pc = full_pc[ii][np.unique(idx[ii])]
            num_unique_idx = np.size(np.unique(idx[ii]))
            remaining_space = k - num_unique_idx
            if remaining_space == 0:
                out_pc[ii] = nn_pc
            else:
                sorted_idx = dist2.argsort()
                remain_sorted_idx = np.setdiff1d(sorted_idx, idx[ii])
                top_idx = remain_sorted_idx[:remaining_space]
                extra_pc = full_pc[ii][top_idx]
                out_pc[ii][0:num_unique_idx][:] = nn_pc
                out_pc[ii][num_unique_idx:k][:] = extra_pc

        return out_pc

    def unique(self, arr):
        _, idx = np.unique(arr, return_index=True)
        return arr[np.sort(idx)]

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def fps_from_given_pc(self, pts, K, given_pc):
        farthest_pts = np.zeros((K, 3))
        t = np.size(given_pc) / 3
        farthest_pts[0:t] = given_pc

        distances = self.calc_distances(farthest_pts[0], pts)
        for i in range(1, t):
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))

        for i in range(t, K):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))
        return farthest_pts

    def fps_from_given_indices(self, pts, K, given_idx):
        farthest_pts = np.zeros((K, 3))
        idx = np.zeros(K, dtype=int)
        t = np.size(given_idx)
        farthest_pts[0:t] = pts[given_idx]
        if t > 1:
            idx[0:t] = given_idx[0:t]
        else:
            idx[0] = given_idx

        distances = self.calc_distances(farthest_pts[0], pts)
        for i in range(1, t):
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))

        for i in range(t, K):
            idx[i] = np.argmax(distances)
            farthest_pts[i] = pts[idx[i]]
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], pts))
        return farthest_pts, idx

    def progressive_forward_projection_plus_continued_fps(self, full_pc, gen_pc, idx):
        batch_size = np.size(full_pc, 0)
        k = np.size(gen_pc, 1)
        out_pc = np.zeros_like(gen_pc)
        for ii in range(0, batch_size):
            i = 1
            best_idx = np.array([], dtype=int)
            while i <= k:
                best_idx = np.append(best_idx, idx[ii, :i])
                best_idx = self.unique(best_idx)
                _, best_idx = self.fps_from_given_indices(full_pc[ii], i, best_idx)
                i = i * 2
            out_pc[ii] = full_pc[ii][best_idx]
        return out_pc

    def simple_projection_and_continued_fps(self, full_pc, gen_pc, idx):
        batch_size = np.size(full_pc, 0)
        k = np.size(gen_pc, 1)
        out_pc = np.zeros_like(gen_pc)
        out_pc_idx = np.zeros([batch_size, k], dtype=int)
        n_unique_points = np.zeros([batch_size, 1])
        for ii in range(0, batch_size):
            best_idx = idx[ii]
            best_idx = self.unique(best_idx)
            n_unique_points[ii] = np.size(best_idx, 0)
            out_pc[ii], out_pc_idx[ii] = self.fps_from_given_indices(full_pc[ii], k, best_idx)
        return out_pc, out_pc_idx, n_unique_points

