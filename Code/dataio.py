from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from utils import *
from tools import *
import config

from icecream import ic


class MetaDataset(Dataset):
    def __init__(self, dataset, var, timesteps, dims=[64, 64, 64], s=4, split='train'):
        # self.v_dir = v_dir
        # self.v_paths = getFilePathsInDir(self.v_dir)#[:3]  if split=='train' else getFilePathsInDir(self.v_dir)[50:52]
        self.root_var_dir = config.root_data_dir + dataset + '/' + var + '/'
        self.v_paths = sorted([self.root_var_dir + f"{dataset}-{var}-{i}.raw" for i in timesteps],key=lambda x: int(os.path.splitext(os.path.split(x)[-1])[0].split('-')[-1]))
        self.v_dataset = self.get_volumes(self.v_paths)
        self.total_coords = torch.tensor(get_mgrid(dims, dim=3, s=1), dtype=torch.float32)
        self.subsample_indices = self.Space_Subsample(dims, s)
        self.v_sub_dataset = torch.zeros((self.subsample_indices.shape[0], self.v_dataset.shape[1]),
                                         dtype=torch.float32)
        self.coords = self.total_coords[self.subsample_indices]
        for i in range(self.v_dataset.shape[1]):
            self.v_sub_dataset[:, i] = self.v_dataset[:, i][self.subsample_indices]
        self.dims = dims

    def Space_Subsample(self, dim, s):
        space_sampled_indices = []
        for z in range(0, dim[2], s):
            for y in range(0, dim[1], s):
                for x in range(0, dim[0], s):
                    index = (((z) * dim[1] + y) * dim[0] + x)
                    space_sampled_indices.append(index)
        space_sampled_indices = np.asarray(space_sampled_indices)
        return space_sampled_indices

    def get_volumes(self, paths):
        volumes = []
        for path in paths:
            v = readDat(path)
            v = ((v - v.min()) / (v.max() - v.min()) - 0.5) * 2
            volumes.append(torch.tensor(v, dtype=torch.float32).unsqueeze(-1))
        volumes = torch.cat(volumes, dim=-1)
        return volumes

    def __len__(self):
        return len(self.v_paths)

    def __getitem__(self, idx):
        sel_volume = self.v_sub_dataset[:, idx]
        indices = torch.randperm(self.coords.shape[0])[:self.coords.shape[0]]
        support_indices = indices[:indices.shape[0] // 2]
        query_indices = indices[indices.shape[0] // 2:]

        meta_dict = {'context': {'x': self.coords[support_indices], 'y': sel_volume[support_indices]},
                     'query': {'x': self.coords[query_indices], 'y': sel_volume[query_indices]},
                     'all': {'x': self.coords, 'y': sel_volume},
                     'total': {'x': self.total_coords, 'y': self.v_dataset[:, idx]},
                     'volume_idx': idx}
        return meta_dict
