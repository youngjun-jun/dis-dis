# README

## Prerequisites

Prepare the following datasets in the specified format:

- **cars3d**
- **shapes3d**
- **mpi3d**
- **celeba**

Place the images in the directory structure:  
`code/datasets/dataset_name/image_{number}.png`

## Training

To start the training process, run the following script:

```bash
/code/scripts/train.sh
```

## Evaluation

To evaluate the trained model, use the following script:

```bash
/code/scripts/eval.sh
```

## Latent Interchange

For latent interchange operations, run the script:

```bash
/code/scripts/latent-interchange.sh
```

## Image Generation for FID

To generate images for the Fréchet Inception Distance (FID) score calculation, use:

```bash
/code/scripts/sampling-for-fid.sh
```


This codebase is based on the paper "DisDiff: Unsupervised Disentanglement of Diffusion Probabilistic Models" presented at NeurIPS 2023. For more details, please refer to the original repository: https://github.com/ThomasMrY/DisDiff