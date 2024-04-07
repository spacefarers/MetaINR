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

loss_threshold = 0.001
PSNR_threshold = 37

heads = [model.SIREN(in_features=3, out_features=1, init_features=64, num_res=3).to(config.device)]

def dict_to_gpu(ob):
    if isinstance(ob, Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    else:
        return ob.cuda()

def evaluate(meta_model:model.MetaModel, head_ind, split_total_coords, meta_batch, step):
    head_learner = meta_model.heads[head_ind]

    sample = dict_to_gpu(meta_batch)
    optimizer = torch.optim.Adam(head_learner.parameters(), lr=5e-5)
    loss = 0.0
    for i in range(meta_model.eval_steps):
        optimizer.zero_grad()
        preds = head_learner(sample['all']['x'])
        loss = loss_func(preds, sample['all']['y'].unsqueeze(-1))
        # learner.adapt(loss)
        loss.backward()
        optimizer.step()
    v_res = []
    for inf_coords in split_total_coords:
        inf_coords = inf_coords.to(config.device)
        preds = head_learner(inf_coords).detach().squeeze().cpu().numpy()
        v_res += list(preds)
    v_res = np.asarray(v_res, dtype=np.float32)
    y_vals = meta_batch['total']['y'].squeeze().numpy()
    GT_range = y_vals.max()-y_vals.min()
    MSE = np.mean((v_res-y_vals)**2)
    PSNR = 20*np.log10(GT_range)-10*np.log10(MSE)
    # saveDat(v_res, f"{config.temp_dir}/preds{step:04d}.raw")
    return PSNR, loss

def save_models(meta_model:model.MetaModel, serial=1):
    torch.save(meta_model.backbone, f"{config.temp_dir}/backbone_{serial}.pth")
    torch.save(meta_model.heads, f"{config.temp_dir}/head_{serial}.pth")
    with open(f"{config.temp_dir}/head_frame_{serial}.json", "w") as f:
        json.dump(meta_model.frame_head_correspondence, f)

def load_models(meta_model:model.MetaModel, serial=1):
    meta_model.backbone = torch.load(f"{config.temp_dir}/backbone_{serial}.pth")
    meta_model.heads = torch.load(f"{config.temp_dir}/head_{serial}.pth")
    if os.path.exists(f"{config.temp_dir}/head_frame_{serial}.json"):
        with open(f"{config.temp_dir}/head_frame_{serial}.json", "r") as f:
            meta_model.frame_head_correspondence = json.load(f)

def run(pretrain=False, serial=1, run_id=1,replay=False):
    config.run_id = run_id
    meta_model = model.MetaModel()
    if pretrain:
        pretrain_model(meta_model, serial)
        return
    load_models(meta_model, 1)
    # heads = [model.Head(backbone).to(config.device)]
    #
    dataset = MetaDataset(config.target_dataset, config.target_var, config.test_timesteps,
                            dims=config.get_dims_of_dataset(config.target_dataset),
                            s=4)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    split_total_coords = torch.split(dataset.total_coords, 32000, dim=0)

    start_time = time.time()

    PSNR_list = []
    for step, meta_batch in enumerate(tqdm(dataloader, desc="Online", leave=False)):
        PSNR, loss = evaluate(meta_model,-1, split_total_coords, meta_batch, step)
        print(f"Preliminary PSNR: {PSNR} Loss: {loss}")
        if PSNR < PSNR_threshold:
            print(colored(f"PSNR: {PSNR} Loss {loss}, Adding new Head #{len(meta_model.heads)}", "red"))
            train_new_head(meta_model, range(max(1,step-2), min(config.test_timesteps[-1],step+2)), replay)
            PSNR, loss = evaluate(meta_model,-1, split_total_coords, meta_batch, step)
        config.log({f"PSNR": PSNR, "loss": loss})
        meta_model.frame_head_correspondence[step] = len(meta_model.heads)-1
        PSNR_list.append(PSNR)

    end_time = time.time()
    print(colored(f"Online Time: {end_time-start_time}", "red"))
    print(colored(f"Average PSNR: {np.mean(PSNR_list)}", "red"))
    print("PSNR_list: ", PSNR_list)

if __name__ == "__main__":
    fire.Fire(run)