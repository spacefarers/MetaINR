from torch.utils.data import DataLoader, Dataset
import config
from dataio import MetaDataset
import learn2learn as l2l
from model import Backbone, Head
from CoordNet import SIREN
import torch
from main import dict_to_gpu
import numpy as np
from tqdm import tqdm
from copy import deepcopy

bundle_size = 80
backbone_group_size = 5
eval_steps = 128
outer_steps = 50
inner_steps = 16

net = SIREN(in_features=3, out_features=1, init_features=64, num_res=3)
head = l2l.algorithms.MAML(net, lr=1e-3, first_order=False).to(config.device)
backbones = []
replay_buffer = []

def add_backbone():
    backbones.append(l2l.algorithms.MAML(Backbone().to(config.device), lr=1e-3, first_order=False, allow_unused=True))


loss_func = torch.nn.MSELoss()
def train_head(time_step):
    global head
    dataset = MetaDataset(config.target_dataset, config.target_var, range(time_step-bundle_size+1, time_step+1),
                            dims=config.get_dims_of_dataset(config.target_dataset),
                            s=4)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    dataloader = list(dataloader)
    # backbone = head.backbone
    meta_optimizer = torch.optim.Adam(head.parameters(), lr=5e-5)
    pbar = tqdm(total=outer_steps)
    for _ in range(outer_steps):
        loss = 0.0
        time_steps = list(range(time_step-bundle_size, time_step))
        np.random.shuffle(time_steps)
        for ind, step in enumerate(time_steps):
            meta_batch = dataloader[step]
            effective_batch_size = meta_batch['context']['x'].shape[0]
            sample = dict_to_gpu(meta_batch)
            meta_optimizer.zero_grad()
            step_loss = 0.0
            for i in range(effective_batch_size):
                learner = head.clone()
                # backbone_learner = backbone.clone()
                # learner.module.backbone = backbone_learner
                for _ in range(inner_steps):
                    support_preds = learner(sample['context']['x'][i])
                    support_loss = loss_func(support_preds, sample['context']['y'][i].unsqueeze(-1))
                    learner.adapt(support_loss)
                    # backbone_learner.adapt(support_loss)
                adapt_loss = loss_func(learner(sample['query']['x'][i]), sample['query']['y'][i].unsqueeze(-1))
                step_loss += adapt_loss
            step_loss = step_loss/effective_batch_size
            step_loss.backward()
            meta_optimizer.step()
            loss += step_loss.item()
            pbar.set_description(f"Step: {ind+1} Loss: {loss/(ind+1)}")
        pbar.update(1)
        # pbar.set_description(f"Loss: {loss/len(dataloader)}")
        config.log({"loss": loss/len(dataloader)})
def evaluate(timestep, meta_batch, split_total_coords):
    global head
    learner = deepcopy(head)
    # head = head.clone()

    # backbone = heads[timestep-bundle_size-1].module.backbone
    # head.module.backbone = backbone
    sample = dict_to_gpu(meta_batch)
    optimizer = torch.optim.Adam(learner.parameters(), lr=1e-4)
    for i in range(eval_steps):
        optimizer.zero_grad()
        preds = learner(sample['all']['x'])
        loss = loss_func(preds, sample['all']['y'].unsqueeze(-1))
        # learner.adapt(loss)
        loss.backward()
        optimizer.step()
    v_res = []
    for inf_coords in split_total_coords:
        inf_coords = inf_coords.to(config.device)
        preds = learner(inf_coords).detach().squeeze().cpu().numpy()
        v_res += list(preds)
    v_res = np.asarray(v_res, dtype=np.float32)
    y_vals = meta_batch['total']['y'].squeeze().numpy()
    GT_range = y_vals.max()-y_vals.min()
    MSE = np.mean((v_res-y_vals)**2)
    PSNR = 20*np.log10(GT_range)-10*np.log10(MSE)
    # saveDat(v_res, f"/mnt/d/tmp/preds{step:04d}.raw")
    return PSNR



def run():
    global head
    # add_backbone()
    # backbones = torch.load("backbone.pth")
    # head = torch.load("head.pth")
    # backbones.append(backbone)
    # heads[0] = head
    # heads[0].module.backbone = backbones[-1]
    # train_head(80)
    # torch.save(head, "head.pth")
    # torch.save(backbones, "backbone.pth")
    dataset = MetaDataset(config.target_dataset, config.target_var, config.test_timesteps,
                            dims=config.get_dims_of_dataset(config.target_dataset),
                            s=4)
    dataloader = list(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0))
    split_total_coords = torch.split(dataset.total_coords, 32000, dim=0)
    # evaluate all timesteps
    PSNR_list = []
    for step, meta_batch in tqdm(enumerate(dataloader), desc="Inferring", leave=False):
        PSNR = evaluate(step, meta_batch, split_total_coords)
        PSNR_list.append(PSNR)
        print("PSNR: ", PSNR)


    # evaluation
    # pbar = tqdm(total=len(dataset))
    # PSNR_list = []
    # for steps,meta_batch in enumerate(dataloader):
    #     sample = dict_to_gpu(meta_batch)
    #     learner = head.clone()
    #     for i in range(eval_steps):
    #         preds = learner(sample['all']['x'])
    #         loss = loss_func(preds, sample['all']['y'].unsqueeze(-1))
    #         learner.adapt(loss)
    #     # inference
    #     v_res = []
    #     for inf_coords in split_total_coords:
    #         inf_coords = inf_coords.to(config.device)
    #         preds = learner(inf_coords).detach().squeeze().cpu().numpy()
    #         v_res += list(preds)
    #     v_res = np.asarray(v_res, dtype=np.float32)
    #     # saveDat(v_res, f"./results/preds{steps:04d}.raw")
    #     v_res = np.asarray(v_res, dtype=np.float32)
    #     y_vals = meta_batch['total']['y'].squeeze().numpy()
    #     GT_range = y_vals.max()-y_vals.min()
    #     MSE = np.mean((v_res-y_vals)**2)
    #     PSNR = 20*np.log10(GT_range)-10*np.log10(MSE)
    #     config.log({"PSNR": PSNR})
    #     PSNR_list.append(PSNR)
    #     pbar.update(1)
    #     pbar.set_description(f"volume time step: {steps}")

    print("PSNR: ", np.mean(PSNR_list))
    print("PSNR list: ", PSNR_list)
    # for step in range(len(dataloader)-bundle_size):
    #     meta_batch = dataloader[step]
    #     print(meta_batch)

if __name__ == "__main__":
    run()