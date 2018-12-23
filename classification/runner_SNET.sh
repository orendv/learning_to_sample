#!/bin/bash

#train PointNet classifier:
python train_classifier.py --log_dir log/baseline/PointNet1024
wait

#train S-NET, use PointNet classifier as the task network:
python train_SNET.py --classifier_model_path log/baseline/PointNet1024/model.ckpt --num_out_points 64 --log_dir log/SNET64
wait

#infer S-NET and evaluate the PointNet classifier over S-NET's sampled points:
python evaluate_SNET.py --sampler_model_path log/SNET64/model.ckpt --dump_dir log/SNET64/eval

#see results in log/SNET64/eval/log_evaluate.txt