dataset_name = ''

if dataset_name == 'cars3d':

    import os
    import numpy as np
    import PIL
    from cars3d import Cars3D

    def save_images_as_png(dataset, save_dir=""):
        """Save images from the dataset as PNG files.

        Args:
            dataset: Cars3D instance.
            save_dir: Directory to save the images.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for i, img in enumerate(dataset.images):
            img = PIL.Image.fromarray((img * 255).astype(np.uint8))
            img.save(os.path.join(save_dir, f"image_{i}.png"))

    dataset = Cars3D()
    save_images_as_png(dataset, save_dir="code/datasets/cars3d")

elif dataset_name == 'shapes3d':

    import h5py
    import os
    from PIL import Image

    h5_file_path = 'shapes3d.h5'
    output_folder = 'code/datasets/shapes3d'
    os.makedirs(output_folder, exist_ok=True)

    with h5py.File(h5_file_path, 'r') as data:
        images = data['images'][:]

    for idx, image_array in enumerate(images):
        output_file_path = os.path.join(output_folder, f'image_{idx}.png')
        
        image = Image.fromarray(image_array)
        image.save(output_file_path)

    print(f'Images saved to {output_folder}')

elif dataset_name == 'mpi3d-toy':

    import numpy as np
    import os
    from PIL import Image

    npz_file_path = 'mpi3d_toy.npz'
    output_folder = 'code/datasets/mpi3d_toy'
    os.makedirs(output_folder, exist_ok=True)

    data = np.load(npz_file_path)
    images = data['images']

    for idx, image_array in enumerate(images):
        output_file_path = os.path.join(output_folder, f'image_{idx}.png')
        
        image = Image.fromarray(image_array)
        image.save(output_file_path)

    print(f'Images saved to {output_folder}')