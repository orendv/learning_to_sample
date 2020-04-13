#!/bin/bash

# train Autoencoder model:
python autoencoder/train_ae.py --train_folder log/autoencoder
wait

# evaluate Autoencoder model
python autoencoder/evaluate_ae.py --train_folder log/autoencoder
wait

# train S-NET, use Autoencoder model as the task network:
python sampler/train_s_net.py --ae_folder log/autoencoder --n_sample_points 64 --train_folder log/s_net_64
wait

# evaluate S-NET:
python sampler/evaluate_s_net.py --train_folder log/s_net_64

# see the results in log/s_net_64/eval/eval_stats_test_set_multi_0064.txt