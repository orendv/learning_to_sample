#!/bin/bash

#train PointNet (vanilla) classifier:
python train_classifier.py --model pointnet_cls_basic --log_dir log/baseline/PointNetVanilla1024
wait

#train ProgressiveNet, use PointNet (vanilla) classifier as the task network:
python train_progressiveNet.py --classifier_model pointnet_cls_basic --classifier_model_path log/baseline/PointNetVanilla1024/model.ckpt --log_dir log/ProgressiveNet
wait

#infer ProgressiveNet and save the ordered point clouds to .h5 files:
python infer_progressiveNet.py --sampler_model_path log/ProgressiveNet/model.ckpt
wait

#evaluate the PointNet (vanilla) classifier using ProgressiveNet's sampled points:
python evaluate_from_files.py --classifier_model pointnet_cls_basic --classifier_model_path log/baseline/PointNetVanilla1024/model.ckpt --data_path log/ProgressiveNet/sampled --dump_dir log/ProgressiveNet/eval

#see results in dump_ProgressiveNet/log_evaluate.txt