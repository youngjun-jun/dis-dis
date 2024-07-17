#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

log_dir='log_dir'
epoch=0

python generation.py \
    --resume $log_dir/checkpoints/epoch=$epoch.ckpt \
    -n 50000 \
    -e 0.0 \
    -l $log_dir \
    -c 100 \
    --batch_size 2000 \
    --config $log_dir/configs/project.yaml \
    --npz $log_dir/dis_repre/epoch=$epoch.npz
