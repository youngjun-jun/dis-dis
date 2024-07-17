#!/bin/bash
# {cars3d, shapes3d, mpi3d, celeba}

export CUDA_VISIBLE_DEVICES=0

python main.py \
    --base configs/latent-diffusion/config.yaml \
    -t \
    --gpus 0 \
    -dt Z \
    -dw 0.05 \
    -l exp/log_dir \
    -s 42 \
    --scale_lr False