from torch.utils.data import Dataset
from torchvision import transforms
import h5py
import numpy as np
import scipy.io as sio
from PIL import Image
from sklearn.utils.extmath import cartesian
import os
import lmdb
from io import BytesIO
import torchvision.transforms.functional as Ftrans

class Shapes3D(Dataset):
    """
    Dataset class for loading images from a folder.
    """
    def __init__(self,
                 path,
                 original_resolution=64,
                 split=None,
                 as_tensor: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        self.path = path
        self.image_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.startswith('image_') and f.endswith('.png')],
                                  key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))
        self.length = len(self.image_files)

        if split is None:
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = []
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        img_path = self.image_files[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img).permute(1, 2, 0)
        return {'image': img}
    
class MPI3D(Dataset):
    """
    Dataset class for loading images from a folder.
    """
    def __init__(self,
                 path,
                 original_resolution=64,
                 split=None,
                 as_tensor: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        self.path = path
        self.image_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.startswith('image_') and f.endswith('.png')],
                                  key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))
        self.length = len(self.image_files)

        if split is None:
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = []
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        img_path = self.image_files[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img).permute(1, 2, 0)
        return {'image': img}

class Cars3D(Dataset):
    """
    Dataset class for loading images from a folder.
    """
    def __init__(self,
                 path,
                 original_resolution=64,
                 split=None,
                 as_tensor: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        self.path = path
        self.image_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.startswith('image_') and f.endswith('.png')],
                                  key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1]))
        self.length = len(self.image_files)

        if split is None:
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = []
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        img_path = self.image_files[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img).permute(1, 2, 0)
        return {'image': img}


class Celeba(Dataset):
    """
    also supports for d2c crop.
    """
    def __init__(self,
                 path,
                 original_resolution=64,
                 split=None,
                 as_tensor: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        # self.data = BaseLMDB(path, original_resolution, zfill=7)
        data = np.load(os.path.join(path,'celeba_64.npz'))
        self.data = data["images"]
        self.length = len(self.data)

        if split is None:
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = []
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img).permute(1, 2, 0)
        return {'image': img}


        
class Shapes3DTrain(Shapes3D):
    def __init__(self, **kwargs):
        # super().__init__(path='../../../guided-diffusion/datasets/',
        super().__init__(path='shapes3d',
                original_resolution=None,
                **kwargs)


class MPI3DTrain(MPI3D):
    def __init__(self, **kwargs):
        super().__init__(path='mpi3d_toy',
                original_resolution=None,
                **kwargs)


class Cars3DTrain(Cars3D):
    def __init__(self, **kwargs):
        super().__init__(path='cars3d',
                original_resolution=None,
                **kwargs)

class CelebaTrain(Celeba):
    def __init__(self, **kwargs):
        super().__init__(path='',
                image_size=64,
                original_resolution=None,
                crop_d2c=True,
                **kwargs)
