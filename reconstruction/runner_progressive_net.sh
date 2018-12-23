#!/bin/bash

# train Autoencoder model:
python autoencoder/train_ae.py --train_folder autoencoder
wait

# train ProgressiveNet, use Autoencoder model as the task network:
python sampler/train_progressive_net.py --ae_folder autoencoder --train_folder progressive_net
wait

# evaluate ProgressiveNet:
python sampler/evaluate_progressive_net.py --n_sample_points 64 --train_folder progressive_net
