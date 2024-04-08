import os.path

import config
import model
from torch.utils.data import Dataset, DataLoader
from dataio import *
from collections.abc import Mapping
from torch import nn
from termcolor import colored
import learn2learn as l2l
from copy import deepcopy
import random
import fire

loss_func = nn.MSELoss()

train_iterations = 500
encoding_time = 0.0

head = model.SIREN(in_features=3, out_features=1, init_features=64, num_res=3).to(config.device)

def dict_to_gpu(ob):
    if isinstance(ob, Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    else:
        return ob.cuda()

optimizer = torch.optim.Adam(head.parameters(), lr=5e-5)
def pretrain_model(meta_batch):
    global head,optimizer
    sample = dict_to_gpu(meta_batch)
    for i in range(2000):
        optimizer.zero_grad()
        preds = head(sample['all']['x'])
        loss = loss_func(preds, sample['all']['y'].unsqueeze(-1))
        loss.backward()
        optimizer.step()


def evaluate(split_total_coords, meta_batch, step, count_time=True):
    global encoding_time, optimizer, head
    sample = dict_to_gpu(meta_batch)
    loss = 0.0
    time_start = time.time()
    for i in range(train_iterations):
        optimizer.zero_grad()
        preds = head(sample['all']['x'])
        loss = loss_func(preds, sample['all']['y'].unsqueeze(-1))
        # learner.adapt(loss)
        loss.backward()
        optimizer.step()
    time_end = time.time()
    if count_time:
        encoding_time += time_end-time_start
    v_res = []
    for inf_coords in split_total_coords:
        inf_coords = inf_coords.to(config.device)
        preds = head(inf_coords).detach().squeeze().cpu().numpy()
        v_res += list(preds)
    v_res = np.asarray(v_res, dtype=np.float32)
    y_vals = meta_batch['total']['y'].squeeze().numpy()
    GT_range = y_vals.max()-y_vals.min()
    MSE = np.mean((v_res-y_vals)**2)
    PSNR = 20*np.log10(GT_range)-10*np.log10(MSE)
    # saveDat(v_res, f"{config.temp_dir}/preds{step:04d}.raw")
    return PSNR, loss


def run(run_id=1):
    config.run_id = run_id

    dataset = MetaDataset(config.target_dataset, config.target_var, config.test_timesteps,
                            dims=config.get_dims_of_dataset(config.target_dataset),
                            s=4)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    split_total_coords = torch.split(dataset.total_coords, 32000, dim=0)

    data = iter(dataloader)
    pretrain_model(next(data))
    PSNR_list = []
    for step, meta_batch in enumerate(dataloader):
        PSNR, loss = evaluate(split_total_coords, meta_batch, step)
        PSNR_list.append(PSNR)
        print(f"Step: {step}, PSNR: {PSNR}, Loss: {loss}")
    print("Average PSNR: ", np.mean(PSNR_list))
    print("Total encoding time: ", encoding_time)
    print("PSNR list: ", PSNR_list)

if __name__ == "__main__":
    fire.Fire(run)