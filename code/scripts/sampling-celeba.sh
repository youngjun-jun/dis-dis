#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

log_dir='log_dir'
epoch=0

python conditional_generation.py \
    --resume $log_dir/checkpoints/epoch=$epoch.ckpt \
    -n 100 \
    -e 0.0 \
    -l $log_dir \
    -c 100 \
    --batch_size 1 \
    --config $log_dir/configs/project.yaml \
    --npz $log_dir/dis_repre/epoch=$epoch.npz
