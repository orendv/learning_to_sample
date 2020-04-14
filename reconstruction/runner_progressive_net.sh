#!/bin/bash

# train Autoencoder model:
python autoencoder/train_ae.py --train_folder log/autoencoder
wait

# evaluate Autoencoder model
python autoencoder/evaluate_ae.py --train_folder log/autoencoder
wait

# train ProgressiveNet, use Autoencoder model as the task network:
python sampler/train_progressive_net.py --ae_folder log/autoencoder --n_sample_points 64 --train_folder log/progressive_net
wait

# evaluate ProgressiveNet:
python sampler/evaluate_progressive_net.py --n_sample_points 64 --train_folder log/progressive_net

# see the results in log/progressive_net/eval/eval_stats_test_set_multi_0064.txt
