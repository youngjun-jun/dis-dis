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
    beta_VAE_score = True
    factor_VAE_score = True
    dci_score = True
    MIG_score = True
    Modularity_score = True
    SAP_score = True
    infoMEC_score=True
    total_results_dict = {}
    
    def _representation(x):
        return pca_rep[x]
    
    if beta_VAE_score:
        with gin.unlock_config():
            from evaluation.metrics.beta_vae import compute_beta_vae_sklearn
            gin.bind_parameter("beta_vae_sklearn.batch_size", 64)
            gin.bind_parameter("beta_vae_sklearn.num_train", 10000)
            gin.bind_parameter("beta_vae_sklearn.num_eval", 5000)
        result_dict = compute_beta_vae_sklearn(eval_dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("beta VAE score:" + str(result_dict))
        total_results_dict["beta_VAE" + preflix] = result_dict
        
    if factor_VAE_score:
        with gin.unlock_config():
            from evaluation.metrics.factor_vae import compute_factor_vae
            gin.bind_parameter("factor_vae_score.num_variance_estimate",10000)
            gin.bind_parameter("factor_vae_score.num_train",10000)
            gin.bind_parameter("factor_vae_score.num_eval",5000)
            gin.bind_parameter("factor_vae_score.batch_size",64)
            gin.bind_parameter("prune_dims.threshold",0.05)
        result_dict = compute_factor_vae(eval_dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("factor VAE score:" + str(result_dict))
        total_results_dict["factor_VAE" + preflix] = result_dict
        
    if dci_score:
        from evaluation.metrics.dci import compute_dci
        with gin.unlock_config():
            gin.bind_parameter("dci.num_train", 10000)
            gin.bind_parameter("dci.num_test", 5000)
        result_dict = compute_dci(eval_dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("dci score:" + str(result_dict))
        total_results_dict["dci" + preflix] = result_dict
        
    if MIG_score:
        with gin.unlock_config():
            from evaluation.metrics.mig import compute_mig
            from evaluation.metrics.utils import _histogram_discretize
            gin.bind_parameter("mig.num_train",10000)
            gin.bind_parameter("discretizer.discretizer_fn",_histogram_discretize)
            gin.bind_parameter("discretizer.num_bins",20)
        result_dict = compute_mig(eval_dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("MIG score:" + str(result_dict))
        total_results_dict["MIG" + preflix] = result_dict

    if Modularity_score:
        with gin.unlock_config():
            from evaluation.metrics.modularity_explicitness import compute_modularity_explicitness
            from evaluation.metrics.utils import _histogram_discretize
            gin.bind_parameter("modularity_explicitness.num_train",10000)
            gin.bind_parameter("modularity_explicitness.num_test",5000)
            gin.bind_parameter("discretizer.discretizer_fn",_histogram_discretize)
            gin.bind_parameter("discretizer.num_bins",20)
        result_dict = compute_modularity_explicitness(eval_dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("Modularity:" + str(result_dict))
        total_results_dict["Modularity" + preflix] = result_dict
       
    if SAP_score:
        with gin.unlock_config():
            from evaluation.metrics.sap_score import compute_sap
            gin.bind_parameter("sap_score.num_train",10000)
            gin.bind_parameter("sap_score.num_test",5000)
            gin.bind_parameter("sap_score.continuous_factors",False)
        result_dict = compute_sap(eval_dataset,_representation,random_state=np.random.RandomState(0),artifact_dir=None)
        print("SAP:" + str(result_dict))
        total_results_dict["SAP" + preflix] = result_dict
       
    if infoMEC_score:
        from evaluation.metrics.infomec import compute_infomec
        result_dict = compute_infomec(eval_dataset, _representation, np.random.RandomState(0),discrete_latents=False,test_size=10000)
        print("infoMEC score:" + str(result_dict))
        partial_dict = {key: result_dict[key] for key in ['infom', 'infoc', 'infoe'] if key in result_dict}
        total_results_dict["infoMEC score" + preflix] = partial_dict
        heatmap=True
        if heatmap:
            fig, ax = plt.subplots(figsize=(10, 5))
            nmi = result_dict['nmi']; active_latents = result_dict['active_latents']
            sns.heatmap(
                nmi, ax=ax, annot=True, fmt='.2f', square=True, vmin=0, vmax=1, cbar=False,
                annot_kws={'fontsize': 8},
                xticklabels=[rf'$\mathbf{{z}}_{{{i}}}$' for i in range(nmi.shape[1])],
                yticklabels=[rf'$\mathbf{{x}}_{{{i}}}$' for i in range(nmi.shape[0])],
                rasterized=True
            )

            for i, label in enumerate(ax.get_xticklabels()):
                if active_latents[i] == 0:
                    label.set_color('red')

            fig.tight_layout()
            plt.savefig(metric_folder + f"/{it}_nmi_heatmap.png", bbox_inches='tight')
            plt.close(fig)

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

def plot_metric(path):
    root_folder = '/'.join(path.split('/')[:-1])
    df_metric = pd.read_csv(path)
    columns_of_interest = ['beta_VAE-eval_accuracy',
                           'factor_VAE-eval_accuracy',
                           'dci-disentanglement',
                           'dci-completeness',
                           'MIG-discrete_mig',
                           'Modularity-modularity_score',
                           'SAP-SAP_score',
                           'infoMEC score-infom',
                           'infoMEC score-infoc',
                           'TAD SCORE',
                        #    'Attributes Captured',
                           ]
    plt.figure(figsize=(10, 6))
    for col in columns_of_interest:
        if col in df_metric.columns:
           plt.plot(df_metric['index'], df_metric[col], label=col)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig(root_folder + '/metric.png')
    plt.clf()

def plot_loss(path):
    try:
        loss_csv = pd.read_csv(os.path.join(path, 'testtube/version_0/metrics.csv'))
        
        sns.lineplot(np.array(loss_csv['train/loss_simple_step'].dropna()))
        plt.savefig(path + '/loss.png')
        plt.clf()
    except:
        print('No loss.csv found')
        pass
        
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
        default=0,
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
        elif data_name == 'celeba1':
            import data.ground_truth.celeba as dceleba
            return dceleba.Dataset(np.arange(0,202599))
        elif data_name == "celeba":
            dataset = CustomImageDataset(zip_file='celeba_64.npz',
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
        append_dict_to_csv(save_dict, os.path.join(args.exp_dir,"metrics.csv"))
        
    from ae_utils_exp import tags, aurocs
    def evaluate_tad(dataset, args):
        os.makedirs(os.path.join(args.exp_dir, "results"), exist_ok=True)
        
        _, targ, idx = next(iter(dataset))
        
        data = os.path.join(args.exp_dir, "dis_repre/epoch={:06d}.npz".format(args.epoch))
        data_array = np.load(data)
        data_array = np.nan_to_num(data_array["latents"])
        latents = torch.from_numpy(data_array[idx])
        pca_list = []
        for i in range(latents.shape[1]):
            pca = PCA(n_components=1)
            pca_result = pca.fit_transform(latents[:,i,:].numpy())
            pca_list.append(pca_result)
            
        pca_rep = np.concatenate(pca_list, axis=1)
        pca_rep = np.nan_to_num(pca_rep)
        
        aurocs_all = torch.ones((40, latents.shape[1])) * 0.5
        with torch.no_grad():
            base_rates_all = targ.sum(dim=0)
            base_rates_all = base_rates_all / targ.shape[0]
            out = torch.from_numpy(pca_rep)
            _ma = out.max(dim=0)[0]
            _mi = out.min(dim=0)[0]
            for i in range(40):
                aurocs_all[i] = aurocs(out.clone(), targ, i, _ma, _mi)
            au_result = aurocs_all ; base_rates_raw = base_rates_all ; targ = targ
            base_rates = base_rates_raw.where(base_rates_raw <= 0.5, 1. - base_rates_raw)
        
        max_aur, argmax_aur = torch.max(au_result.clone(), dim=1)
        norm_diffs = torch.zeros(40)
        aurs_diffs = torch.zeros(40)
        for ind, tag, max_a, argmax_a, aurs in zip(range(40), tags, max_aur.clone(), argmax_aur.clone(), au_result.clone()):
            norm_aurs = (aurs.clone() - 0.5) / (aurs.clone()[argmax_a] - 0.5)
            aurs_next = aurs.clone()
            aurs_next[argmax_a] = 0.0
            aurs_diff = max_a - aurs_next.max()
            aurs_diffs[ind] = aurs_diff
            norm_aurs[argmax_a] = 0.0
            norm_diff = 1. - norm_aurs.max()
            norm_diffs[ind] = norm_diff
            print("{}\t\t Lat: {}\t Max: {:1.3f}\t ND: {:1.3f}".format(tag, argmax_a.item(), max_a.item(), norm_diff.item()))
            plt_ind = ind//5, ind%5
            assert aurs.max() == max_a

        with torch.no_grad():
            not_targ = 1 - targ
            j_prob = lambda x, y: torch.logical_and(x, y).sum() / x.numel()
            mi = lambda jp, px, py: 0. if jp == 0. or px == 0. or py == 0. else jp*torch.log(jp/(px*py))

            mi_mat = torch.zeros((40, 40))
            for i in range(40):
                i_mp = targ[:, i].sum() / targ.shape[0]
                for j in range(40):
                    j_mp = targ[:, j].sum() / targ.shape[0]
                    # FF
                    jp = j_prob(not_targ[:, i], not_targ[:, j])
                    pi = 1. - i_mp
                    pj = 1. - j_mp
                    mi_mat[i][j] += mi(jp, pi, pj)
                    # FT
                    jp = j_prob(not_targ[:, i], targ[:, j])
                    pi = 1. - i_mp
                    pj = j_mp
                    mi_mat[i][j] += mi(jp, pi, pj)
                    # TF
                    jp = j_prob(targ[:, i], not_targ[:, j])
                    pi = i_mp
                    pj = 1. - j_mp
                    mi_mat[i][j] += mi(jp, pi, pj)
                    # TT
                    jp = j_prob(targ[:, i], targ[:, j])
                    pi = i_mp
                    pj = j_mp
                    mi_mat[i][j] += mi(jp, pi, pj)

            mi_maxes, mi_inds = (mi_mat * (1 - torch.eye(40))).max(dim=1)
            ent_red_prop = 1. - (mi_mat.diag() - mi_maxes) / mi_mat.diag()
            print(mi_mat.diag())
        thresh = 0.75
        ent_red_thresh = 0.2

        filt = (max_aur >= thresh).logical_and(ent_red_prop <= ent_red_thresh)
        aurs_diffs_filt = aurs_diffs[filt]
        print(len(aurs_diffs_filt))
        plt.figure(figsize=(10, 7))
        plt.ylim((0.0, 1.0))
        plt.title("Total AUROC Diff: {:1.3f} at Thresh: {:1.2f}".format(aurs_diffs_filt.sum(), thresh))
        plt.ylabel("AUROC Difference")
        plt.xlabel("Attribute")
        plt.xticks(rotation=90)
        plt.bar(range(aurs_diffs.shape[0]), aurs_diffs, tick_label=tags)
        plt.grid(which='both', axis='y')
        plt.savefig(os.path.join(args.exp_dir, "results", 'tad_image-{:06d}.jpg'.format(args.epoch)))

        print("TAD SCORE: ", aurs_diffs_filt.sum().item(), "Attributes Captured: ", len(aurs_diffs_filt))
        result_dict = {'TAD SCORE': aurs_diffs_filt.sum().item(), 'Attributes Captured': len(aurs_diffs_filt)}
        
        save_dict = OrderedDict({'index': args.epoch})
        save_dict.update(result_dict)
        append_dict_to_csv(save_dict, os.path.join(args.exp_dir, "metrics.csv"))

    
    dataset = make_dis_dataset(args.dataset, args.tad_samples)
    print(args.dataset)
    plot_loss(args.exp_dir)
    if args.dataset == 'celeba':
        evaluate_tad(dataset,args)
    elif args.dataset in ['shapes3d', 'mpi3d', 'cars3d']:
        evaluate_process(dataset, args)
    else:
        raise NotImplementedError()
    print('Generating Metric plot')
    plot_metric(os.path.join(args.exp_dir,"metrics.csv"))
    plot_loss(args.exp_dir)