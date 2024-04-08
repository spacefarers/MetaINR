# Use "--pretrain" flag to run the pretraining phase
# Use "--replay" flag to enable replay

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
PSNR_threshold = 36
retrain_per = 10 # steps

def dict_to_gpu(ob):
    if isinstance(ob, Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    else:
        return ob.cuda()

def subset_clip(l, max_size=3):
    return random.sample(l, min(max_size,len(l)))


def pretrain_model(meta_model:model.MetaModel, serial=1):
    meta_model.meta_lr = 5e-5 # very sensitive
    meta_model.outer_steps = 500


    dataset1 = MetaDataset(config.target_dataset, config.target_var, range(1,4),
                            dims=config.get_dims_of_dataset(config.target_dataset),
                            s=4)
    dataset2 = MetaDataset(config.target_dataset, config.target_var, range(4,7),
                            dims=config.get_dims_of_dataset(config.target_dataset),
                            s=4)
    dataloader1 = DataLoader(dataset1, batch_size=1, shuffle=True, num_workers=0)
    dataloader2 = DataLoader(dataset2, batch_size=1, shuffle=True, num_workers=0)
    pbar = tqdm(total=meta_model.outer_steps)
    head1 = model.Head().to(config.device)
    head2 = model.Head().to(config.device)
    head_model1 = l2l.algorithms.MAML(head1, lr=1e-4, first_order=True, allow_unused=True).to(config.device)
    head_model2 = l2l.algorithms.MAML(head2, lr=1e-4, first_order=True, allow_unused=True).to(config.device)
    backbone_model = l2l.algorithms.MAML(meta_model.backbone, lr=1e-5, first_order=True, allow_unused=True).to(config.device)
    meta_optimizer = torch.optim.Adam([
        {'params': head_model1.parameters(), 'lr': meta_model.meta_lr},
        {'params': head_model2.parameters(), 'lr': meta_model.meta_lr},
        {'params': backbone_model.parameters(), 'lr': meta_model.meta_lr}
    ])
    for outer_step in range(meta_model.outer_steps):
        loss = 0.0
        data_obj1 = iter(dataloader1)
        data_obj2 = iter(dataloader2)
        for ind in range(3):
            total_loss = 0.0
            sample = dict_to_gpu(next(data_obj1))
            head_model = head_model1
            for j in range(2):
                sample['head'] = j
                if outer_step == 0 and ind == 2:
                    meta_model.replay_buffer.append(sample)
                effective_batch_size = sample['context']['x'].shape[0]
                meta_optimizer.zero_grad()
                step_loss = 0.0
                for i in range(effective_batch_size):
                    learner = head_model.clone()
                    backbone_learner = backbone_model.clone()
                    # optimizer = torch.optim.Adam(learner.parameters(), lr=1e-4)
                    for _ in range(meta_model.inner_steps):
                        # optimizer.zero_grad()
                        support_preds = learner(backbone_learner(sample['context']['x'][i]))
                        support_loss = loss_func(support_preds, sample['context']['y'][i].unsqueeze(-1))
                        # support_loss.backward()
                        # optimizer.step()
                        learner.adapt(support_loss)
                        # backbone_learner.adapt(support_loss)
                    adapt_loss = loss_func(learner(backbone_learner(sample['query']['x'][i])), sample['query']['y'][i].unsqueeze(-1))
                    step_loss += adapt_loss
                total_loss += step_loss/effective_batch_size
                if j == 0:
                    sample = dict_to_gpu(next(data_obj2))
                    head_model = head_model2

            total_loss.backward()
            meta_optimizer.step()
            loss += total_loss.item()
        pbar.set_description(f"Loss: {loss/len(dataloader1)/2}")
        config.log({"loss": loss/len(dataloader1)/2})
        pbar.update(1)
    meta_model.heads = [head1, head2]
    for i in range(7):
        meta_model.frame_head_correspondence[i] = 0 if i < 4 else 1

    # train_new_head(meta_model, range(1, 6))
    save_models(meta_model, serial)
def train_new_head(meta_model:model.MetaModel, time_steps, replay=False):
    dataset = MetaDataset(config.target_dataset, config.target_var, time_steps,
                            dims=config.get_dims_of_dataset(config.target_dataset),
                            s=4)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    pbar = tqdm(total=meta_model.outer_steps)

    head = model.Head().to(config.device)
    if len(meta_model.heads) > 0:
        head.load_state_dict(meta_model.heads[-1].state_dict())
    head_model = l2l.algorithms.MAML(head, lr=1e-4, first_order=True, allow_unused=True).to(config.device)
    backbone_model = l2l.algorithms.MAML(meta_model.backbone, lr=1e-4, first_order=True, allow_unused=True).to(config.device)
    replay_batch = None
    meta_optimizer = torch.optim.Adam([
        {'params': head_model.parameters(), 'lr': meta_model.meta_lr},
        {'params': backbone_model.parameters(), 'lr': meta_model.meta_lr}
    ])
    for outer_step in range(meta_model.outer_steps):
        replay_set = subset_clip(meta_model.replay_buffer,1)
        loss = 0.0
        replay_loss = 0.0
        for ind, meta_batch in enumerate(dataloader):
            sample = dict_to_gpu(meta_batch)
            if ind == len(dataloader)-1:
                sample['head'] = len(meta_model.heads)
                replay_batch = sample
            effective_batch_size = meta_batch['context']['x'].shape[0]
            meta_optimizer.zero_grad()
            step_loss = 0.0
            for i in range(effective_batch_size):
                learner = head_model.clone()
                backbone_learner = backbone_model.clone()
                for _ in range(meta_model.inner_steps):
                    support_preds = learner(backbone_learner(sample['context']['x'][i]))
                    support_loss = loss_func(support_preds, sample['context']['y'][i].unsqueeze(-1))
                    learner.adapt(support_loss)
                adapt_loss = loss_func(learner(backbone_learner(sample['query']['x'][i])), sample['query']['y'][i].unsqueeze(-1))
                step_loss += adapt_loss
            step_loss = step_loss/effective_batch_size

            loss += step_loss.item()
            if replay:
                for sample in replay_set:
                    replay_step_loss = 0.0
                    effective_batch_size = sample['context']['x'].shape[0]
                    replay_head = deepcopy(meta_model.heads[sample['head']])
                    for i in range(effective_batch_size):
                        adapt_loss = loss_func(replay_head(backbone_model(sample['query']['x'][i])), sample['query']['y'][i].unsqueeze(-1))
                        replay_step_loss += adapt_loss
                    replay_step_loss = replay_step_loss/effective_batch_size/len(replay_set)
                    step_loss += replay_step_loss
                    replay_loss += replay_step_loss

            step_loss.backward()
            meta_optimizer.step()
        pbar.set_description(f"Loss: {loss/len(dataloader)}, Replay: {replay_loss/len(dataloader)}")
        pbar.update(1)
    meta_model.heads.append(head)
    if replay:
        meta_model.replay_buffer.append(replay_batch)


    # torch.save(backbones[0], f"{config.temp_dir}/backbone.pth")
    # torch.save(base, f"{config.temp_dir}/base.pth")
    # torch.save(heads[0], f"{config.temp_dir}/head.pth")

def evaluate(meta_model:model.MetaModel, head_ind, split_total_coords, meta_batch, step, count_time=True):
    head_learner = deepcopy(meta_model.heads[head_ind])
    backbone_learner = deepcopy(meta_model.backbone)
    head_learner.backbone = backbone_learner

    sample = dict_to_gpu(meta_batch)
    optimizer = torch.optim.Adam(head_learner.parameters(), lr=5e-5)
    loss = 0.0
    time_start = time.time()
    for i in range(meta_model.eval_steps):
        optimizer.zero_grad()
        preds = head_learner(sample['all']['x'])
        loss = loss_func(preds, sample['all']['y'].unsqueeze(-1))
        # learner.adapt(loss)
        loss.backward()
        optimizer.step()
    time_end = time.time()
    if count_time:
        meta_model.tmp_encode_time += time_end-time_start
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
    torch.save(meta_model, f"{config.temp_dir}/meta_model_{serial}.pth")

def load_models(serial=1)->model.MetaModel:
    meta_model = torch.load(f"{config.temp_dir}/meta_model_{serial}.pth")
    return model.MetaModel().load_in(meta_model.__dict__)


def run(pretrain=False, serial=None, run_id=1,replay=False,dataset=None,var=None):
    config.run_id = run_id
    config.enabled_replay = replay
    if dataset is not None:
        config.target_dataset = dataset
        config.target_var = var if var is not None else "default"
        config.test_timesteps = range(1,config.get_size_of_dataset(config.target_dataset)+1)
    if serial is None:
        serial = config.dataset_to_serial[config.target_dataset]
    if pretrain:
        config.pretraining = True
        meta_model = model.MetaModel()
        pretrain_model(meta_model, serial)
        return
    meta_model = load_models(serial)
    print(f"Model Structure: {meta_model.backbone.layers}+{meta_model.heads[0].layers}")
    meta_model.frame_head_correspondence += [-1]*max(len(config.test_timesteps)-len(meta_model.frame_head_correspondence),0)

    meta_model.meta_lr = 5e-5
    meta_model.outer_steps = 50

    dataset = MetaDataset(config.target_dataset, config.target_var, config.test_timesteps,
                            dims=config.get_dims_of_dataset(config.target_dataset),
                            s=4)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    split_total_coords = torch.split(dataset.total_coords, 32000, dim=0)


    meta_model.online_PSNR_seq = [] # Performing retrain before transmission of current timestep
    meta_model.online_PSNR_par = [] # Performing retrain after transmission

    # Online Main Loop
    retrain_strikes = 0
    for step, meta_batch in enumerate(tqdm(dataloader, desc="Online", leave=False)):
        PSNR, loss = evaluate(meta_model,meta_model.frame_head_correspondence[step], split_total_coords, meta_batch, step)
        tqdm.write(f"Preliminary PSNR {step}: {PSNR} Loss: {loss}")
        meta_model.online_PSNR_seq.append(PSNR)
        if retrain_strikes == retrain_per:
            tqdm.write(colored(f"PSNR: {PSNR} Loss {loss}, Adding new Head #{len(meta_model.heads)}", "red"))
            time_start = time.time()
            train_new_head(meta_model, range(max(1,step-2), min(config.test_timesteps[-1],step+2)), replay)
            time_end = time.time()
            meta_model.tmp_encode_time += time_end-time_start
            PSNR, loss = evaluate(meta_model,-1, split_total_coords, meta_batch, step, count_time=False) # Don't double count retrain evals.
            meta_model.frame_head_correspondence[step-2] = len(meta_model.heads)-1
            meta_model.frame_head_correspondence[step-1] = len(meta_model.heads)-1
            retrain_strikes = 0
        if meta_model.frame_head_correspondence[step] == -1:
            meta_model.frame_head_correspondence[step] = len(meta_model.heads)-1
            retrain_strikes += 1
        config.log({f"PSNR": PSNR, "loss": loss})
        meta_model.online_PSNR_par.append(PSNR)
    online_encode_time = meta_model.tmp_encode_time
    meta_model.tmp_encode_time = 0
    meta_model.online_encode_time = online_encode_time
    print(f"Online Encoding Time: ", online_encode_time)
    print(colored(f"Online Seq PSNR: {np.mean(meta_model.online_PSNR_seq)}", "red"))
    print(colored(f"Online Par PSNR: {np.mean(meta_model.online_PSNR_par)}", "red"))
    print("Online Seq PSNR PSNR_list: ", meta_model.online_PSNR_seq)
    print("Online Par PSNR PSNR_list: ", meta_model.online_PSNR_par)

    # meta_model = load_models(serial+1)
    print("Head Frame Correspondence: ", meta_model.frame_head_correspondence)

    # Head Frame Linked Eval
    meta_model.transfer_PSNR = []
    for step, meta_batch in enumerate(tqdm(dataloader, desc="Transfer Inferring", leave=False)):
        PSNR, loss = evaluate(meta_model,meta_model.frame_head_correspondence[step], split_total_coords, meta_batch, step)
        print(f"Transfer Linked PSNR {step}: {PSNR} Loss: {loss}")
        meta_model.transfer_PSNR.append(PSNR)
    transfer_encode_time = meta_model.tmp_encode_time
    meta_model.tmp_encode_time = 0
    meta_model.transfer_encode_time = transfer_encode_time
    print(f"Transfer Linked Average PSNR: {np.mean(meta_model.transfer_PSNR)}")
    print("Transfer Linked PSNR_list: ", meta_model.transfer_PSNR)
    print("Transfer Encode Time: ", transfer_encode_time)

    # Final Eval
    meta_model.last_frame_PSNR = []
    for step, meta_batch in enumerate(tqdm(dataloader, desc="Final Step Inferring", leave=False)):
        PSNR, loss = evaluate(meta_model,-1, split_total_coords, meta_batch, step)
        meta_model.last_frame_PSNR.append(PSNR)
    print(f"Final Average PSNR: {np.mean(meta_model.last_frame_PSNR)}")
    print("Final PSNR_list: ", meta_model.last_frame_PSNR)
    save_models(meta_model, serial+1)

if __name__ == "__main__":
    fire.Fire(run)