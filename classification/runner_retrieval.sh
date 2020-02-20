#!/bin/bash

#train PointNet classifier (if it does not already exist):
if [ ! -d log/baseline/PointNet1024 ]; then
    python train_classifier.py --log_dir log/baseline/PointNet1024
    wait
fi

#train S-NET (if it does not already exist), use PointNet classifier as the task network:
if [ ! -d log/SNET32 ]; then
    python train_SNET.py --classifier_model_path log/baseline/PointNet1024/model.ckpt --num_out_points 32 --log_dir log/SNET32
    wait
fi

#infer S-NET and evaluate the PointNet classifier over S-NET's sampled points and save data for retrieval:
python evaluate_SNET.py --sampler_model_path log/SNET32/model.ckpt --num_out_points 32 --dump_dir log/SNET32/eval --save_retrieval_vectors 1
wait

#analyze retrieval data:
python analyze_precision_recall.py --num_out_points 32 --dump_dir log/SNET32/retrieval/

#see results in log/SNET32/retrieval/