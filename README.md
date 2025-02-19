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

## Files
### Cars3d
- **Model:** [Download](https://drive.google.com/file/d/1OdTum8mFAUCSfeXxDBipzzlh5PacBbLn/view?usp=drive_link)
- **Gaussians:** [Download](https://drive.google.com/file/d/1BMMH55TC7aFmHZZyxicMo-m5kp72Sd6j/view?usp=drive_link)
- **Label CSV (for Test):** [Download](https://drive.google.com/file/d/1VQo7hTNjwkDYOael-dl4_ZI2U_Qp9-C0/view?usp=drive_link)

### Shapes3d
- **Model:** [Download](https://drive.google.com/file/d/1Zj-6idOErPzFr3YRaAfZ7KuSt6_THgcN/view?usp=drive_link)
- **Gaussians:** [Download](https://drive.google.com/file/d/1zCbr27XhBqCXXO3jRrPPLBcM5g31KJQ5/view?usp=drive_link)
- **Label CSV (for Test):** [Download](https://drive.google.com/file/d/1qo5P1wcuemweMXfwqGyp_M4ZWjKZcqI3/view?usp=drive_link)

### MPI3D-toy
- **Model:** [Download](https://drive.google.com/file/d/1Tq5ROEKzWkuNh05SosCU3wfYuq8ioiX8/view?usp=drive_link)
- **Gaussians:** [Download](https://drive.google.com/file/d/1sWKoYx8uMSdijsmslIueySkY6nvaVKEq/view?usp=drive_link)
- **Label CSV (for Test):** [Download](https://drive.google.com/file/d/120dOm-tBVOaNcpVnWy1B42Q8h8py5BOE/view?usp=drive_link)

### CelebA
- **Model:** [Download](https://drive.google.com/file/d/1kQ-_W_nj7MZkHmEcPoMUegI6nrHyJ-K7/view?usp=drive_link)
- **Gaussians:** [Download](https://drive.google.com/file/d/1FDadeGqzpKMZcjVomJxs1o_xmbVqclR5/view?usp=drive_link)
- **Label CSV (for Test, included in the dataset):** [Download](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) (Kaggle)

## Acknowledgement

We based our codes on [ThomasMrY/DisDiff](https://github.com/ThomasMrY/DisDiff)
