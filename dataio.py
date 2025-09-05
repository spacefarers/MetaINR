from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
from utils import *
from tools import *
from einops import rearrange, repeat
from icecream import ic
import config


class MetaDataset(Dataset):
    def __init__(self, dataset, var, t_range=None, dims=None, s=4, subsample_half=False):
        if dims is None:
            dims = [64, 64, 64]
        dims = config.get_dims_of_dataset(dataset)
        self.v_dir = config.root_data_dir + dataset + '/' + var + '/'
        self.v_paths = getFilePathsInDir(self.v_dir)
        if var != "all":
            self.v_paths = self.v_paths[t_range[0]:t_range[1]]
        if subsample_half:
            self.v_paths = self.v_paths[::2]
        print(self.v_paths)
        self.v_dataset = self.get_volumes(self.v_paths)
        self.total_coords = torch.tensor(get_mgrid(dims, dim=3, s=1), dtype=torch.float32)
        self.subsample_indices = self.Space_Subsample(dims, s)
        self.v_sub_dataset = torch.zeros((self.subsample_indices.shape[0], self.v_dataset.shape[1]), dtype=torch.float32)
        self.coords = self.total_coords[self.subsample_indices]
        for i in range(self.v_dataset.shape[1]):
            self.v_sub_dataset[:, i] = self.v_dataset[:, i][self.subsample_indices]
        self.dims = dims

    def Space_Subsample(self, dim, s):
        space_sampled_indices = []
        for z in range(0, dim[2], s):
            for y in range(0, dim[1], s):
                for x in range(0, dim[0], s):
                    index = (((z)*dim[1]+y)*dim[0]+x)
                    space_sampled_indices.append(index)
        space_sampled_indices = np.asarray(space_sampled_indices)
        return space_sampled_indices

    def get_volumes(self, paths):
        volumes = []
        for path in paths:
            v = readDat(path)
            volumes.append(torch.tensor(v, dtype=torch.float32).unsqueeze(-1))
        volumes = torch.cat(volumes, dim=-1)
        # normalize the data to [-1, 1]
        volumes = (volumes - volumes.min()) / (volumes.max() - volumes.min()) * 2 - 1
        return volumes

    def __len__(self):
        return len(self.v_paths)

    def __getitem__(self, idx):
        sel_volume = self.v_sub_dataset[:, idx]
        indices = torch.randperm(self.coords.shape[0])[:self.coords.shape[0]]
        support_indices = indices[:indices.shape[0]//2]
        query_indices = indices[indices.shape[0]//2:]

        meta_dict = {'context': {'x': self.coords[support_indices], 'y': sel_volume[support_indices]},
                     'total': {'x': self.total_coords, 'y': self.v_dataset[:, idx]},
                     'query': {'x': self.coords[query_indices], 'y': sel_volume[query_indices]},
                     'all': {'x': self.coords, 'y': sel_volume},
                     'volume_idx': idx}
        return meta_dict

# * reference from: https://cs330.stanford.edu/materials/cs330_multitask_transfer_2023.pdf (page 20)
class PretrainDataset(Dataset):
    def __init__(self, dataset, var, t_range, s=4, split='train'):
        dims = config.get_dims_of_dataset(dataset)
        self.v_dir = config.root_data_dir + dataset + '/' + var + '/'
        self.split = split
        self.v_paths = getFilePathsInDir(self.v_dir)[t_range[0]:t_range[1]:2] if split == 'train' else getFilePathsInDir(self.v_dir)[t_range[0]:t_range[1]]
        self.dims = dims
        self.s = s
        print(self.v_paths)
        self.dataset = self.get_TrainDataSet()

    def get_TrainDataSet(self):
        self.v_dataset = self.get_volumes(self.v_paths)
        self.total_coords = torch.tensor(get_mgrid(self.dims, dim=3, s=1), dtype=torch.float32)
        self.subsample_indices = self.Space_Subsample(self.dims, self.s)
        self.v_sub_dataset = torch.zeros((self.subsample_indices.shape[0], self.v_dataset.shape[1]), dtype=torch.float32)  # 32768,5
        self.coords = self.total_coords[self.subsample_indices]
        for i in range(self.v_dataset.shape[1]):
            self.v_sub_dataset[:, i] = self.v_dataset[:, i][self.subsample_indices]
        train_coords = (repeat(self.coords, 'n d -> n t d', t=self.v_dataset.shape[1])) #.reshape(-1, 3)
        train_values = self.v_sub_dataset.unsqueeze(-1) #.reshape(-1, 1)  # 32768*5, 1
        train_data = torch.cat([train_coords, train_values], dim=-1)
        return train_data

    def Space_Subsample(self, dim, s):
        space_sampled_indices = []
        for z in range(0, dim[2], s):
            for y in range(0, dim[1], s):
                for x in range(0, dim[0], s):
                    index = (((z)*dim[1]+y)*dim[0]+x)
                    space_sampled_indices.append(index)
        space_sampled_indices = np.asarray(space_sampled_indices)
        return space_sampled_indices

    def get_volumes(self, paths):
        volumes = []
        for path in paths:
            v = readDat(path)
            volumes.append(torch.tensor(v, dtype=torch.float32).unsqueeze(-1))
        volumes = torch.cat(volumes, dim=-1)
        return volumes

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return self.dataset[idx, :, :]


if __name__ == "__main__":
    pretrain_dataset = PretrainDataset(v_dir="/mnt/d/data/vorts/default",
                                       dims=[128, 128, 128],
                                       t_range=(0, 10),
                                       s=4)
    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=1, shuffle=True, num_workers=0)
    exit()
