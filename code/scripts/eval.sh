#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
python evaluate_renew.py \
    --exp_dir exp/log_dir \
    --dataset mpi3d # {cars3d, shapes3d, mpi3d, celeba}
