# Disentangling Disentangled Representations: Towards Improved Latent Units via Diffusion Models

## Prepare data
* Cars3d   
Download [nips2015-analogy-data.tar.gz](http://www.scottreed.info/files/nips2015-analogy-data.tar.gz)
* Shapes3d  
Download [3dshapes.h5](https://console.cloud.google.com/storage/browser/3d-shapes)
* MPI3D-toy
Download [mpi3d_toy.npz](https://storage.googleapis.com/mpi3d_disentanglement_dataset/data/mpi3d_toy.npz)
* CelebA
Download [CelebA/Img](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (CelebA)
or
Download [celeba-dataset.zip](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) (Kaggle)

Prepare the following datasets in the specified format:

- **cars3d**
- **shapes3d**
- **mpi3d**
- **celeba**

```bash
code/datasets/dataset.py
```

Place the images in the directory structure:  
`code/datasets/dataset_name/image_{number}.png`

## Training

To start the training process, run the following script:

```bash
code/scripts/train.sh
```

## Evaluation

To evaluate the trained model, use the following script:

```bash
code/scripts/eval.sh
```

## Latent Interchange

For latent interchange operations, run the script:

```bash
code/scripts/latent-interchange.sh
```

## Image Generation for FID

To generate images for the Fréchet Inception Distance (FID) score calculation, use:

```bash
code/scripts/sampling-for-fid.sh
```

## Acknowledgement

We based our codes on [ThomasMrY/DisDiff](https://github.com/ThomasMrY/DisDiff)