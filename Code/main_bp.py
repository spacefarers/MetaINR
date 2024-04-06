import config
import model
from torch.utils.data import Dataset, DataLoader
from dataio import *
from collections.abc import Mapping
from torch import nn
from termcolor import colored
import learn2learn as l2l
from copy import deepcopy

loss_func = nn.MSELoss()

meta_lr = 5e-5
outer_steps = 500
inner_steps = 16
eval_steps = 100
loss_threshold = 0.001
PSNR_threshold = 37

def dict_to_gpu(ob):
    if isinstance(ob, Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    else:
        return ob.cuda()

def train_new_head(meta_model:model.MetaModel, time_steps):
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
    meta_optimizer = torch.optim.Adam(list(head_model.parameters())+list(backbone_model.parameters()), lr=meta_lr)
    replay_batch = None
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
        for meta_batch in meta_model.replay_buffer:
            effective_batch_size = meta_batch['context']['x'].shape[0]
            sample = dict_to_gpu(meta_batch)
            meta_optimizer.zero_grad()
            step_loss = 0.0
            for i in range(effective_batch_size):
                learner = head_model.clone()
                for _ in range(inner_steps):
                    support_preds = learner(sample['context']['x'][i])
                    support_loss = loss_func(support_preds, sample['context']['y'][i].unsqueeze(-1))
                    learner.adapt(support_loss)
                adapt_loss = loss_func(learner(sample['query']['x'][i]), sample['query']['y'][i].unsqueeze(-1))
                step_loss += adapt_loss
            step_loss = step_loss/effective_batch_size
            step_loss.backward()
            meta_optimizer.step()
            loss += step_loss.item()
        pbar.update(1)
        pbar.set_description(f"Loss: {loss/(len(dataloader)+len(meta_model.replay_buffer))}")
    meta_model.heads.append(head)
    meta_model.replay_buffer.append(replay_batch)


    # torch.save(backbones[0], f"/mnt/d/tmp/metaINR/backbone.pth")
    # torch.save(base, f"/mnt/d/tmp/metaINR/base.pth")
    # torch.save(heads[0], f"/mnt/d/tmp/metaINR/head.pth")

def evaluate(meta_model:model.MetaModel, head_ind, split_total_coords, meta_batch, step):
    head_learner = deepcopy(meta_model.heads[head_ind])
    backbone_learner = deepcopy(meta_model.backbone)
    head_learner.backbone = backbone_learner

    sample = dict_to_gpu(meta_batch)
    optimizer = torch.optim.Adam(list(head_learner.parameters())+list(backbone_learner.parameters()), lr=5e-5)
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
    # saveDat(v_res, f"/mnt/d/tmp/preds{step:04d}.raw")
    return PSNR, loss

def save_models(meta_model:model.MetaModel, serial=1):
    torch.save(meta_model.backbone, f"/mnt/d/tmp/metaINR/backbone_{serial}.pth")
    torch.save(meta_model.heads, f"/mnt/d/tmp/metaINR/head_{serial}.pth")

def load_models(meta_model:model.MetaModel, serial=1):
    meta_model.backbone = torch.load(f"/mnt/d/tmp/metaINR/backbone_{serial}.pth")
    meta_model.heads = torch.load(f"/mnt/d/tmp/metaINR/head_{serial}.pth")

def run():
    meta_model = model.MetaModel()

    train_new_head(meta_model, range(1, 6))
    save_models(meta_model, 1)
    return
    #
    load_models(meta_model, 1)
    # heads = [model.Head(backbone).to(config.device)]
    #
    dataset = MetaDataset(config.target_dataset, config.target_var, config.test_timesteps,
                            dims=config.get_dims_of_dataset(config.target_dataset),
                            s=4)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    split_total_coords = torch.split(dataset.total_coords, 32000, dim=0)
    # #
    start_time = time.time()

    PSNR_list = []
    for step, meta_batch in enumerate(tqdm(dataloader, desc="Online", leave=False)):
        PSNR, loss = evaluate(meta_model,-1, split_total_coords, meta_batch, step)
        config.log({f"{step} PSNR": PSNR, "loss": loss})
        if loss > 1e-4:
            print(colored(f"PSNR: {PSNR} Loss {loss}, Adding new Head #{len(meta_model.heads)}", "red"))
            train_new_head(meta_model, range(max(1,step+1), min(config.test_timesteps[-1],step+6)))
            PSNR, loss = evaluate(meta_model,-1, split_total_coords, meta_batch, step)
            config.log({f"New {step} PSNR": PSNR, "New loss": loss})
        meta_model.frame_head_correspondence[step] = len(meta_model.heads)-1
        PSNR_list.append(PSNR)

    end_time = time.time()
    print(f"Online Time: {end_time-start_time}")
    print(f"Average PSNR: {np.mean(PSNR_list)}")
    print("PSNR_list: ", PSNR_list)
    save_models(meta_model, 2)

    # load_models(2)

    # Final Eval
    PSNR_list = []
    for step, meta_batch in enumerate(tqdm(dataloader, desc="Inferring", leave=False)):
        PSNR, loss = evaluate(meta_model,meta_model.frame_head_correspondence[step], split_total_coords, meta_batch, step)
        # config.log({"PSNR": PSNR, "loss": loss})
        PSNR_list.append(PSNR)
    print(f"Final Average PSNR: {np.mean(PSNR_list)}")
    print("Final PSNR_list: ", PSNR_list)
    print("Head Frame Correspondence: ", meta_model.frame_head_correspondence)

if __name__ == "__main__":
    run()