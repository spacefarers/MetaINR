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

meta_lr = 1e-4
outer_steps = 150
inner_steps = 16
eval_steps = 250
loss_threshold = 0.001
PSNR_threshold = 40

def dict_to_gpu(ob):
    if isinstance(ob, Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    else:
        return ob.cuda()

def subset_clip(l, max_size=3):
    if len(l) > max_size:
        return random.sample(l, max_size)
    return l

def train_new_head(meta_model:model.MetaModel, time_steps, replay=False):
    dataset = MetaDataset(config.target_dataset, config.target_var, time_steps,
                            dims=config.get_dims_of_dataset(config.target_dataset),
                            s=4)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    pbar = tqdm(total=outer_steps)
    total_coords = dataset.total_coords
    split_total_coords = torch.split(total_coords, 32000, dim=0)

    head = model.Head().to(config.device)
    if len(meta_model.heads) > 0:
        head.load_state_dict(meta_model.heads[-1].state_dict())
    head_model = l2l.algorithms.MAML(head, lr=1e-3, first_order=True, allow_unused=True).to(config.device)
    backbone_model = l2l.algorithms.MAML(meta_model.backbone, lr=1e-3, first_order=True, allow_unused=True).to(config.device)
    replay_batch = None
    meta_optimizer = torch.optim.Adam([
        {'params': head_model.parameters(), 'lr': meta_lr},
        {'params': backbone_model.parameters(), 'lr': meta_lr}
    ])
    replay_optimizer = torch.optim.Adam(backbone_model.parameters(), lr=1e-5)
    for outer_step in range(outer_steps):
        total_loss = 0.0
        mean_PSNR = 0.0
        loss = 0.0
        for ind, meta_batch in enumerate(dataloader):
            if replay_batch is None:
                replay_batch = meta_batch
            effective_batch_size = meta_batch['context']['x'].shape[0]
            sample = dict_to_gpu(meta_batch)
            meta_optimizer.zero_grad()
            step_loss = 0.0
            for i in range(effective_batch_size):
                learner = head_model.clone()
                backbone_learner = backbone_model.clone()
                # optimizer = torch.optim.Adam(learner.parameters(), lr=1e-4)
                for _ in range(inner_steps):
                    # optimizer.zero_grad()
                    support_preds = learner(backbone_learner(sample['context']['x'][i]))
                    support_loss = loss_func(support_preds, sample['context']['y'][i].unsqueeze(-1))
                    # support_loss.backward()
                    # optimizer.step()
                    learner.adapt(support_loss)
                    # backbone_learner.adapt(support_loss)
                adapt_loss = loss_func(learner(backbone_learner(sample['query']['x'][i])), sample['query']['y'][i].unsqueeze(-1))
                step_loss += adapt_loss
            step_loss = step_loss/effective_batch_size
            step_loss.backward()
            meta_optimizer.step()
            loss += step_loss.item()
        replay_set = subset_clip(meta_model.replay_buffer)
        for meta_batch in replay_set:
            effective_batch_size = meta_batch['context']['x'].shape[0]
            sample = dict_to_gpu(meta_batch)
            replay_optimizer.zero_grad()
            step_loss = 0.0
            for i in range(effective_batch_size):
                learner = head_model.clone()
                backbone_learner = backbone_model.clone()
                # optimizer = torch.optim.Adam(learner.parameters(), lr=1e-4)
                for _ in range(inner_steps):
                    # optimizer.zero_grad()
                    support_preds = learner(backbone_learner(sample['context']['x'][i]))
                    support_loss = loss_func(support_preds, sample['context']['y'][i].unsqueeze(-1))
                    # support_loss.backward()
                    # optimizer.step()
                    learner.adapt(support_loss)
                    # backbone_learner.adapt(support_loss)
                adapt_loss = loss_func(learner(backbone_learner(sample['query']['x'][i])), sample['query']['y'][i].unsqueeze(-1))
                step_loss += adapt_loss
            step_loss = step_loss/effective_batch_size
            step_loss.backward()
            replay_optimizer.step()
            loss += step_loss.item()
        pbar.set_description(f"Loss: {loss/(len(dataloader)+len(replay_set))}")
        pbar.update(1)
    meta_model.heads.append(head)
    if replay:
        meta_model.replay_buffer.append(replay_batch)


    # torch.save(backbones[0], f"{config.temp_dir}/backbone.pth")
    # torch.save(base, f"{config.temp_dir}/base.pth")
    # torch.save(heads[0], f"{config.temp_dir}/head.pth")

def evaluate(meta_model:model.MetaModel, head_ind, split_total_coords, meta_batch, step):
    head_learner = deepcopy(meta_model.heads[head_ind])
    backbone_learner = deepcopy(meta_model.backbone)
    head_learner.backbone = backbone_learner

    sample = dict_to_gpu(meta_batch)
    optimizer = torch.optim.Adam(head_learner.parameters(), lr=5e-5)
    loss = 0.0
    for i in range(eval_steps):
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
    # save_models(meta_model, 2)

    # load_models(meta_model, 2)

    # Final Eval
    PSNR_list = []
    for step, meta_batch in enumerate(tqdm(dataloader, desc="Inferring", leave=False)):
        # PSNR, loss = evaluate(meta_model,meta_model.frame_head_correspondence[step], split_total_coords, meta_batch, step)
        PSNR, loss = evaluate(meta_model,-1, split_total_coords, meta_batch, step)
        # config.log({"PSNR": PSNR, "loss": loss})
        PSNR_list.append(PSNR)
    print(f"Final Average PSNR: {np.mean(PSNR_list)}")
    print("Final PSNR_list: ", PSNR_list)
    print("Head Frame Correspondence: ", meta_model.frame_head_correspondence)

def pretrain_model(meta_model:model.MetaModel, serial=1):
    global meta_lr, outer_steps
    meta_lr = 5e-5
    outer_steps = 500
    train_new_head(meta_model, range(1, 6))
    save_models(meta_model, serial)

if __name__ == "__main__":
    fire.Fire(run)