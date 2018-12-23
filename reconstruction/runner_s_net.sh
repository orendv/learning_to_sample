#!/bin/bash

# train Autoencoder model:
python autoencoder/train_ae.py --train_folder autoencoder
wait

# train S-NET, use Autoencoder model as the task network:
python sampler/train_s_net.py --ae_folder autoencoder --n_sample_points 64 --train_folder s_net_64
wait

# evaluate S-NET:
python sampler/evaluate_s_net.py --train_folder s_net_64
