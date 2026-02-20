# Disentangling Disentangled Representations: Towards Improved Latent Units via Diffusion Models
[Paper](https://openaccess.thecvf.com/content/WACV2025/html/Jun_Disentangling_Disentangled_Representations_Towards_Improved_Latent_Units_via_Diffusion_Models_WACV_2025_paper.html)

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
- **First Stage Model:** [Download](https://drive.google.com/file/d/1fdc7oI_n1RgWk7QYi0JDSJQDWhopCNXh/view?usp=sharing) (Drive)
- **Model:** [Download](https://drive.google.com/file/d/1gzv9E2QqiVy5DzeaJEk_o6Q2QPaHXuh9/view?usp=sharing) (Drive)
- **Representations:** [Download](https://drive.google.com/file/d/1TJIEKFVxpjEzr_Q08rh03AwCmFueQ6ht/view?usp=sharing) (Drive)
- **Gaussians:** [Download](https://drive.google.com/file/d/1BYoLu4-IRo5598p6vLNEkMmOBZzltber/view?usp=sharing) (Drive)
- **Label CSV (for Test):** [Download](https://drive.google.com/file/d/1ndOAnFaZedGfbqp4MnkFFCwilRVmcwnM/view?usp=sharing) (Drive)

### Shapes3d
- **First Stage Model:** [Download](https://drive.google.com/file/d/1uKRJUcY6Jk77T1izxtS9cbniQ1c3C0s1/view?usp=sharing) (Drive)
- **Model:** [Download](https://drive.google.com/file/d/1EAu5TJo1KlZIBUSWldIirUSJQvYRHccm/view?usp=sharing) (Drive)
- **Representations:** [Download](https://drive.google.com/file/d/15kFDoKH1m0hyG6j4pzFhLg1zKaMJwrG9/view?usp=sharing) (Drive)
- **Gaussians:** [Download](https://drive.google.com/file/d/1cvm0tG64igcidr8D6NzbHdRhA2JHYg0E/view?usp=sharing) (Drive)
- **Label CSV (for Test):** [Download](https://drive.google.com/file/d/1sbtn5rGVkGzq37WykTIWszKEBLCFLjFp/view?usp=sharing) (Drive)

### MPI3D-toy
- **First Stage Model:** [Download](https://drive.google.com/file/d/1_5xyLxdpBdQI6fNgPEbHVZ0JPxBqZlQf/view?usp=sharing) (Drive)
- **Model:** [Download](https://drive.google.com/file/d/1PCIo-79x0Lw2u6DPMlWFoZKiKeWGiKRz/view?usp=sharing) (Drive)
- **Representations:** [Download](https://drive.google.com/file/d/1HIwk8FbfN9m6p7-FB_OhJ37gGRfdRuak/view?usp=sharing) (Drive)
- **Gaussians:** [Download](https://drive.google.com/file/d/1-TFps1mxnqlTcOhm0lxsfk6F02B4EDFA/view?usp=sharing) (Drive)
- **Label CSV (for Test):** [Download](https://drive.google.com/file/d/1qP4NY-7zc6MH5iJ5gGvkqhKbfx6ZG_sS/view?usp=sharing) (Drive)

### CelebA
- **First Stage Model:** [Download](https://drive.google.com/file/d/1-rtCLgyCNj9m4jbFUlVTnKPV2ubI8lx0/view?usp=sharing) (Drive)
- **Model:** [Download](https://drive.google.com/file/d/1mU8ToGR9Mdcvx-klXl1KlNs5HJa9f1mD/view?usp=sharing) (Drive)
- **Representations:** [Download](https://drive.google.com/file/d/1vT9eb073AHi_wy4ZLDgDeQzlIYUvjZCn/view?usp=sharing) (Drive)
- **Gaussians:** [Download](https://drive.google.com/file/d/1VCgtg0XRHUHDDMdvV8fmafzU5nufLUtV/view?usp=sharing) (Drive)
- **Label CSV (for Test, included in the dataset):** [Download](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) (Kaggle)

## Acknowledgement

We based our codes on [ThomasMrY/DisDiff](https://github.com/ThomasMrY/DisDiff)

