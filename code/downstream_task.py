from sklearn.decomposition import PCA
import numpy as np
import torch
import gin
import os
import json
import argparse
import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_text(result_dict,file):
    with open(file,'w+') as f:
        json.dump(result_dict,f)

def eval_func(eval_dataset, data_tensor, metric_folder, it, preflix=""):

    pca_list = []
    for i in range(data_tensor.shape[1]):
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(data_tensor[:,i,:].numpy())
        pca_list.append(pca_result)
    pca_rep = np.concatenate(pca_list, axis=1)
    pca_rep = np.nan_to_num(pca_rep)
    total_results_dict = {}
    
    downstream= True
    
    def _representation(x):
        return pca_rep[x]
    
    
    if downstream:
        with gin.unlock_config():
            from evaluation.metrics.downstream_task import compute_downstream_task
            from evaluation.metrics.utils import gradient_boosting_classifier
            gin.bind_parameter('predictor.predictor_fn', gradient_boosting_classifier)
            gin.bind_parameter('downstream_task.num_train', [10000,1000,100])
            gin.bind_parameter('downstream_task.num_test', 5000)
        result_dict = compute_downstream_task(eval_dataset, _representation, np.random.RandomState(0))
        only_efficiency = True
        if only_efficiency:
            result_dict = {k: v for k,v in result_dict.items() if k in ('1000:sample_efficiency','100:sample_efficiency')}
        print('Sample efficiency_1000:' + str(result_dict['1000:sample_efficiency']) + ', Sample efficiency_100:' + str(result_dict['100:sample_efficiency']))
        total_results_dict['Sample_efficiency'] = result_dict
    
        
    return total_results_dict        
            
def find_metric_in_jsons(folder_path,metric,sub):
    metrics = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                json_data = json.load(file)
                json_data = json_data[metric][sub]
                metrics.append(json_data)
    return metrics

        
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    def __init__(self, zip_file, csv_file):
        self.image_df = np.load(zip_file)['images']
        self.labels_df = pd.read_csv(csv_file)
        transform = []
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)
        
    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.image_df[idx]
        label = torch.from_numpy(np.array(self.labels_df.iloc[idx,1:],dtype=int))
        image = self.transform(image).permute(1, 2, 0)
        return image, label, idx
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="shapes3d",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--tad_samples",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    
    args = parser.parse_args()
    set_seed(args.seed)

    def make_dis_dataset(data_name, tad_samples):
        if data_name == 'shapes3d':
            import data.ground_truth.shapes3d as dshapes3d
            return dshapes3d.Dataset(np.arange(0,480000))
        elif data_name == "mpi3d":
            import data.ground_truth.mpi3d as dmpi3d
            return dmpi3d.Dataset(np.arange(0,1036800))
        elif data_name == "cars3d":
            import data.ground_truth.cars3d as dcars3d
            return dcars3d.Dataset(np.arange(0,17568))
        elif data_name == "celeba":
            dataset = CustomImageDataset(zip_file='datasets/celeba_64.npz',
                                         csv_file='celeba/attr_celeba.csv')
            eval_bs = tad_samples
            dataloader = DataLoader(dataset, batch_size=eval_bs, shuffle=True, num_workers=4)
            return dataloader
        else:
            raise NotImplementedError()
        
    def append_dict_to_csv(data, filename):
        fieldnames = {}
        row = {}
        
        for key, sub_dict in data.items():
            if isinstance(sub_dict, dict):
                for sub_key, value in sub_dict.items():
                    field_key = f"{key}-{sub_key}"
                    if field_key not in fieldnames:
                        fieldnames[field_key] = None
                    row[field_key] = value
            else:
                if key not in fieldnames:
                    fieldnames[key] = None
                row[key] = sub_dict
        
        try:
            df_existing = pd.read_csv(filename)
            for field in fieldnames:
                if field not in df_existing.columns:
                    df_existing[field] = None
        except FileNotFoundError:
            df_existing = pd.DataFrame(columns=fieldnames)
        
        df_existing = df_existing.dropna(axis=1, how='all')
        
        df_new = pd.DataFrame([row], columns=fieldnames)
        df_new = df_new.dropna(axis=1, how='all')
        
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(filename, index=False)

    
    from collections import OrderedDict

    def evaluate_process(label_dataset, args):
        os.makedirs(os.path.join(args.exp_dir, "results"), exist_ok=True)
        
            
        data = os.path.join(args.exp_dir, "dis_repre/epoch={:06d}.npz".format(args.epoch))
        data_array = np.load(data)
        data_array = np.nan_to_num(data_array["latents"])
        latents = torch.from_numpy(data_array)
        result_dict = eval_func(label_dataset, latents, os.path.join(args.exp_dir, "results"), "results-{:06d}".format(args.epoch))
        
        save_dict = OrderedDict({'index': args.epoch})
        save_dict.update(result_dict)
        append_dict_to_csv(save_dict, os.path.join(args.exp_dir,"metrics_down.csv"))


    

    
    dataset = make_dis_dataset(args.dataset, args.tad_samples)
    print(args.dataset)
    
    if args.dataset in ['shapes3d', 'mpi3d', 'cars3d']:
        evaluate_process(dataset, args)
