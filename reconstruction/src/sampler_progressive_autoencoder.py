'''
Created on September 7th, 2018

@author: itailang
'''

import warnings
import os.path as osp
import tensorflow as tf
import numpy as np

from tflearn import is_training

from .in_out import create_dir, pickle_data, unpickle_data
from .general_utils import apply_augmentations, iterate_in_chunks
from .neural_net import Neural_Net, MODEL_SAVER_ID


class SamplerProgressiveAutoEncoder(Neural_Net):
    '''Basis class for a Neural Network that implements a Sampler with Auto-Encoder in TensorFlow.
    '''

    def __init__(self, name, graph, configuration):
        Neural_Net.__init__(self, name, graph)
        self.is_denoising = configuration.is_denoising
        self.n_input = configuration.n_input
        self.n_output = configuration.n_output

        in_shape = [None] + self.n_input
        out_shape = [None] + self.n_output

        samp_shape = [None] + configuration.n_samp

        with tf.variable_scope(name):
            self.x = tf.placeholder(tf.float32, in_shape)
            if self.is_denoising:
                self.gt = tf.placeholder(tf.float32, out_shape)
            else:
                self.gt = self.x

            # for samples interpolation
            self.s1 = tf.placeholder(tf.float32, samp_shape)
            self.s2 = tf.placeholder(tf.float32, samp_shape)

            if self.is_denoising:
                self.pc1 = tf.placeholder(tf.float32, out_shape)
                self.pc2 = tf.placeholder(tf.float32, out_shape)

    def restore_ae_model(self, ae_model_path, ae_name, epoch, verbose=False):
        '''Restore all the variables of a saved ae model.
        '''
        global_vars = tf.global_variables()
        ae_params = [v for v in global_vars if v.name.startswith(ae_name)]

        saver_ae = tf.train.Saver(var_list=ae_params)
        saver_ae.restore(self.sess, osp.join(ae_model_path, MODEL_SAVER_ID + '-' + str(int(epoch))))

        if verbose:
            print('AE Model restored from %s, in epoch %d' % (ae_model_path, epoch))


    def partial_fit(self, X, GT=None, S=None, compute_recon=False):
        '''Trains the model with mini-batches of input data.
        If GT is not None, then the reconstruction loss compares the output of the net that is fed X, with the GT.
        This can be useful when training for instance a denoising auto-encoder.
        Returns:
            The loss of the mini-batch.
            The reconstructed (output) point-clouds.
        '''
        if compute_recon:
            x_reconstr = self.x_reconstr
        else:
            x_reconstr = self.no_op

        is_training(True, session=self.sess)
        try:
            if GT is None:
                feed_dict = {self.x: X}
            else:
                feed_dict = {self.x: X, self.gt: GT}

            if S is not None:
                feed_dict[self.s] = S

            _, loss, loss_ae, loss_similarity, recon = self.sess.run(
                (self.train_step, self.loss, self.loss_ae, self.loss_similarity, x_reconstr), feed_dict=feed_dict)

            is_training(False, session=self.sess)
        except Exception:
            raise
        finally:
            is_training(False, session=self.sess)
        return recon, loss, loss_ae, loss_similarity

    def reconstruct(self, X, GT=None, S=None, compute_loss=True):
        '''Use AE to reconstruct given data.
        GT will be used to measure the loss (e.g., if X is a noisy version of the GT)'''
        if compute_loss:
            loss = self.loss
            loss_ae = self.loss_ae
            loss_similarity = self.loss_similarity
        else:
            loss = self.no_op
            loss_ae = self.no_op
            loss_similarity = self.no_op

        feed_dict = {self.x: X}
        if GT is not None:
            feed_dict[self.gt] = GT

        if S is not None:
            feed_dict[self.s] = S

        return self.sess.run((self.x_reconstr, loss, loss_ae, loss_similarity), feed_dict=feed_dict)

    def sort(self, X):
        '''Sort points by farthest point sampling indices.'''
        return self.sess.run(self.x_sorted, feed_dict={self.x: X})

    def sample(self, X):
        '''Sample points from input data.'''
        generated_points, idx = self.sess.run((self.s, self.idx), feed_dict={self.x: X})
        sampled_points, sampled_points_idx, n_unique_points = self.simple_projection_and_continued_fps(X, generated_points, idx)

        return generated_points, sampled_points, sampled_points_idx, n_unique_points

    def match_samples(self, S1, S2):
        return self.sess.run(self.s1_matched, feed_dict={self.s1: S1, self.s2: S2})

    def get_match_cost(self, PC1, PC2):
        return self.sess.run(self.match_cost, feed_dict={self.pc1: PC1, self.pc2: PC2})

    def get_nn_distance(self, X, S):
        return self.sess.run(self.nn_distance, feed_dict={self.x: X, self.s: S})

    def transform(self, X):
        '''Transform data by mapping it into the latent space.'''
        return self.sess.run(self.z, feed_dict={self.x: X})

    def interpolate(self, x, y, steps):
        ''' Interpolate between and x and y input vectors in latent space.
        x, y np.arrays of size (n_points, dim_embedding).
        '''
        in_feed = np.vstack((x, y))
        z1, z2 = self.transform(in_feed.reshape([2] + self.n_input))
        all_z = np.zeros((steps + 2, len(z1)))

        for i, alpha in enumerate(np.linspace(0, 1, steps + 2)):
            all_z[i, :] = (alpha * z2) + ((1.0 - alpha) * z1)

        return self.sess.run((self.x_reconstr), {self.z: all_z})

    def interpolate_samples(self, s1, s2, steps):
        ''' Interpolate between and s1 and s1 samples.
        s1, s2 are np.arrays of size (n_samp_points, 3).
        '''
        s1_matched = self.match_samples(np.expand_dims(s1, axis=0), np.expand_dims(s2, axis=0))
        s1_matched = np.squeeze(s1_matched, axis=0)

        all_s = np.zeros([steps + 2] + [len(s1)] + [3])
        for i, alpha in enumerate(np.linspace(0, 1, steps + 2)):
            all_s[i, :, :] = (alpha * s2) + ((1.0 - alpha) * s1_matched)

        return all_s, self.sess.run(self.x_reconstr, {self.s: all_s})

    def decode(self, z):
        if np.ndim(z) == 1:  # single example
            z = np.expand_dims(z, 0)
        return self.sess.run((self.x_reconstr), {self.z: z})

    def train(self, train_data, configuration, log_file=None, held_out_data=None):
        c = configuration
        stats = []

        if c.saver_step is not None:
            create_dir(c.train_dir)

        for _ in xrange(c.training_epochs):
            loss, loss_ae, loss_similarity, duration = self._single_epoch_train(train_data, c)
            epoch = int(self.sess.run(self.increment_epoch))
            stats.append((epoch, loss, loss_ae, loss_similarity, duration))

            if epoch % c.loss_display_step == 0:
                print("Epoch:", '%04d' % (epoch), 'training time (minutes)=', "{:.4f}".format(duration / 60.0), "loss=",
                      "{:.9f}".format(loss), "loss_ae=", "{:.9f}".format(loss_ae), "loss_similarity=", "{:.9f}".format(loss_similarity))
                if log_file is not None:
                    log_file.write('%04d\t%.9f\t%.9f\t%.9f\t%.4f\n' % (epoch, loss, loss_ae, loss_similarity, duration / 60.0))

            # Save the models checkpoint periodically.
            if c.saver_step is not None and (epoch % c.saver_step == 0 or epoch - 1 == 0):
                checkpoint_path = osp.join(c.train_dir, MODEL_SAVER_ID)
                self.saver.save(self.sess, checkpoint_path, global_step=self.epoch)

            if c.exists_and_is_not_none('summary_step') and (epoch % c.summary_step == 0 or epoch - 1 == 0):
                summary = self.sess.run(self.merged_summaries)
                self.train_writer.add_summary(summary, epoch)

            if held_out_data is not None and c.exists_and_is_not_none('held_out_step') and (
                    epoch % c.held_out_step == 0):
                loss, loss_ae, loss_similarity, duration = self._single_epoch_train(held_out_data, c, only_fw=True)
                print("Held Out Data :", 'forward time (minutes)=', "{:.4f}".format(duration / 60.0), "loss=",
                      "{:.9f}".format(loss), "loss_ae=", "{:.9f}".format(loss_ae), "loss_similarity=", "{:.9f}".format(loss_similarity))
                if log_file is not None:
                    log_file.write('On Held_Out: %04d\t%.9f\t%.9f\t%.9f\t%.4f\n' % (epoch, loss, loss_ae, loss_similarity, duration / 60.0))
        return stats

    def evaluate(self, in_data, configuration, samp_gen=None, samp_prj=None, samp_prj_idx=None, n_unique_prj=None, ret_pre_augmentation=False):
        n_examples = in_data.num_examples
        b = configuration.batch_size

        # perform augmentations
        pre_aug = None
        if self.is_denoising:
            original_data, ids, feed_data = in_data.full_epoch_data(shuffle=False)
            if ret_pre_augmentation:
                pre_aug = feed_data.copy()
            if feed_data is None:
                feed_data = original_data
            feed_data = apply_augmentations(feed_data, configuration)  # This is a new copy of the batch.
        else:
            original_data, ids, _ = in_data.full_epoch_data(shuffle=False)
            feed_data = apply_augmentations(original_data, configuration)

        # sample points
        if (samp_gen is None) and (samp_prj is None):
            samp_gen, samp_prj, samp_prj_idx, n_unique_prj = self.get_samples(feed_data, b)

        # sort the input points by FPS indices
        feed_data = self.get_sorted_data(feed_data, b)

        n_sample = samp_gen.shape[1]
        samp_fps = feed_data[:, :n_sample, :]

        # compute chamfer distance between sample and input point cloud
        chamfer_dist_gen = np.squeeze(self.get_nn_distances(feed_data, samp_gen, b))
        chamfer_dist_prj = np.squeeze(self.get_nn_distances(feed_data, samp_prj, b))
        chamfer_dist_fps = np.squeeze(self.get_nn_distances(feed_data, samp_fps, b))

        # reconstruct
        recon_gen = np.zeros([n_examples] + self.n_output)
        recon_prj = np.zeros([n_examples] + self.n_output)
        recon_fps = np.zeros([n_examples] + self.n_output)
        for i in xrange(0, n_examples, b):
            if self.is_denoising:
                recon_gen[i:i + b] = self.reconstruct(feed_data[i:i + b], original_data[i:i + b], S=samp_gen[i:i + b], compute_loss=True)[0]
                recon_prj[i:i + b] = self.reconstruct(feed_data[i:i + b], original_data[i:i + b], S=samp_prj[i:i + b], compute_loss=True)[0]
                recon_fps[i:i + b] = self.reconstruct(feed_data[i:i + b], original_data[i:i + b], S=samp_fps[i:i + b], compute_loss=True)[0]
            else:
                recon_prj[i:i + b] = self.reconstruct(feed_data[i:i + b], S=samp_prj[i:i + b], compute_loss=False)[0]
                recon_gen[i:i + b] = self.reconstruct(feed_data[i:i + b], S=samp_gen[i:i + b], compute_loss=False)[0]
                recon_fps[i:i + b] = self.reconstruct(feed_data[i:i + b], S=samp_fps[i:i + b], compute_loss=False)[0]

        # Compute reconstruction loss per point cloud
        ae_loss_gen = np.zeros(n_examples)
        ae_loss_prj = np.zeros(n_examples)
        ae_loss_fps = np.zeros(n_examples)
        for i in xrange(0, n_examples, 1):
            if self.is_denoising:
                ae_loss_gen[i] = self.sess.run(self.loss_ae, feed_dict={self.gt: original_data[i:i+1], self.s: samp_gen[i:i+1]})
                ae_loss_prj[i] = self.sess.run(self.loss_ae, feed_dict={self.gt: original_data[i:i+1], self.s: samp_prj[i:i+1]})
                ae_loss_fps[i] = self.sess.run(self.loss_ae, feed_dict={self.gt: original_data[i:i+1], self.s: samp_fps[i:i+1]})
            else:
                ae_loss_gen[i] = self.sess.run(self.loss_ae, feed_dict={self.x: feed_data[i:i+1], self.s: samp_gen[i:i+1]})
                ae_loss_prj[i] = self.sess.run(self.loss_ae, feed_dict={self.x: feed_data[i:i+1], self.s: samp_prj[i:i+1]})
                ae_loss_fps[i] = self.sess.run(self.loss_ae, feed_dict={self.x: feed_data[i:i+1], self.s: samp_fps[i:i+1]})


        ret = {'feed_data': feed_data, 'ids': ids, 'original_data': original_data, 'pre_aug': pre_aug,
               'samp_gen': samp_gen, 'samp_prj': samp_prj, 'samp_fps': samp_fps,
               'chamfer_dist_gen': chamfer_dist_gen, 'chamfer_dist_prj': chamfer_dist_prj, 'chamfer_dist_fps': chamfer_dist_fps,
               'samp_prj_idx': samp_prj_idx, 'n_unique_prj': n_unique_prj,
               'recon_gen': recon_gen, 'recon_prj': recon_prj, 'recon_fps': recon_fps,
               'ae_loss_gen': ae_loss_gen, 'ae_loss_prj': ae_loss_prj, 'ae_loss_fps': ae_loss_fps}

        return ret

    def embedding_at_tensor(self, dataset, conf, feed_original=True, apply_augmentation=False,
                            tensor_name='bottleneck'):
        '''
        Observation: the NN-neighborhoods seem more reasonable when we do not apply the augmentation.
        Observation: the next layer after latent (z) might be something interesting.
        tensor_name: e.g. model.name + '_1/decoder_fc_0/BiasAdd:0'
        '''
        batch_size = conf.batch_size
        original, ids, noise = dataset.full_epoch_data(shuffle=False)

        if feed_original:
            feed = original
        else:
            feed = noise
            if feed is None:
                feed = original

        feed_data = feed
        if apply_augmentation:
            feed_data = apply_augmentations(feed, conf)

        embedding = []
        if tensor_name == 'bottleneck':
            for b in iterate_in_chunks(feed_data, batch_size):
                embedding.append(self.transform(b.reshape([len(b)] + conf.n_input)))
        else:
            embedding_tensor = self.graph.get_tensor_by_name(tensor_name)
            for b in iterate_in_chunks(feed_data, batch_size):
                codes = self.sess.run(embedding_tensor, feed_dict={self.x: b.reshape([len(b)] + conf.n_input)})
                embedding.append(codes)

        embedding = np.vstack(embedding)
        return feed, embedding, ids

    def get_latent_codes(self, pclouds, batch_size=100):
        ''' Convenience wrapper of self.transform to get the latent (bottle-neck) codes for a set of input point
        clouds.
        Args:
            pclouds (N, K, 3) numpy array of N point clouds with K points each.
        '''
        latent_codes = []
        idx = np.arange(len(pclouds))
        for b in iterate_in_chunks(idx, batch_size):
            latent_codes.append(self.transform(pclouds[b]))
        return np.vstack(latent_codes)

    def get_samples(self, pclouds, batch_size=100):
        ''' Convenience wrapper of self.sample to get the samples for a set of input point clouds.
        Args:
            pclouds (N, K, 3) numpy array of N point clouds with K points each.
            batch_size size of point clouds batch
        '''
        generated_samples = []
        projected_samples = []
        projected_samples_idx = []
        num_unique_points = []
        idx = np.arange(len(pclouds))
        for b in iterate_in_chunks(idx, batch_size):
            gen, prj, prj_idx, n_unique_prj = self.sample(pclouds[b])
            generated_samples.append(gen)
            projected_samples.append(prj)
            projected_samples_idx.append(prj_idx)
            num_unique_points.append(n_unique_prj)
        return np.vstack(generated_samples), np.vstack(projected_samples), np.vstack(projected_samples_idx), np.squeeze(np.vstack(num_unique_points))

    def get_nn_distances(self, pclouds, samples, batch_size=100):
        ''' Convenience wrapper of self.get_nn_distance to get the chamfer distance between point clouds and samples
        Args:
            pclouds (N, K, 3) numpy array of N point clouds with K points each.
            samples (N, Ks, 3) numpy array of N point clouds with Ks points each, Ks <= K.
        '''
        nn_distances = []
        idx = np.arange(len(pclouds))
        for b in iterate_in_chunks(idx, batch_size):
            nn_distances.append(self.get_nn_distance(pclouds[b], samples[b]))
        return np.vstack(nn_distances)

    def get_sorted_data(self, in_data, batch_size):
        n_examples = np.size(in_data, 0)
        b = batch_size

        # sort by FPS indices
        sorted_data = np.zeros(in_data.shape)
        for i in xrange(0, n_examples, b):
            sorted_data[i:i + b] = self.sort(in_data[i:i + b])

        return sorted_data

    def get_matches_cost(self, pclouds, pclouds_noisy):
        num_example = pclouds.shape[0]
        num_example_noisy = pclouds_noisy.shape[0]
        num_points = pclouds.shape[1]
        matches_cost = np.zeros([num_example, num_example_noisy])
        for i in xrange(num_example):
            for j in xrange(num_example_noisy):
                print "matching example %d/%d to noisy example %d/%d" % (i+1, num_example, j+1, num_example_noisy)
                matches_cost[i, j] = np.squeeze(self.get_match_cost(pclouds[i:i+1], pclouds_noisy[j:j+1]), axis=0)/num_points

        return matches_cost
