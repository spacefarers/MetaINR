import config
import model
from torch.utils.data import Dataset, DataLoader
from dataio import *
from collections.abc import Mapping
from torch import nn
from termcolor import colored
import learn2learn as l2l

base = model.Base()
backbones = []
heads = [model.Head().to(config.device) for _ in config.test_timesteps]
replay_buffer = []
loss_func = nn.MSELoss()

meta_lr = 1e-4  # very sensitive
fast_lr = 0.001
outer_steps = 6
inner_steps = 200
eval_steps = 500
PSNR_threshold = 35

def dict_to_gpu(ob):
    if isinstance(ob, Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    else:
        return ob.cuda()

def add_backbone():
    backbones.append(model.Backbone(base).to(config.device))
    model.weights_init_kaiming(backbones[-1])
    # base.backbones.append(backbones[-1])

def train_backbone(time_steps):
    dataset = MetaDataset(config.target_dataset, config.target_var, time_steps,
                            dims=config.get_dims_of_dataset(config.target_dataset),
                            s=4)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    pbar = tqdm(total=outer_steps)
    total_coords = dataset.total_coords
    split_total_coords = torch.split(total_coords, 32000, dim=0)
    # backbone = l2l.algorithms.MAML(backbones[-1], lr=1e-4, first_order=False, allow_nograd=True)
    # meta_optimizer = torch.optim.Adam(backbone.parameters(), lr=meta_lr)
    dataloader = list(dataloader)
    time_steps = list(time_steps)
    # shuffle

    replay_meta_batch = replay_head = None
    for outer_step in range(outer_steps):
        total_loss = 0.0
        mean_PSNR = 0.0
        # backbone_clones = [backbone.clone() for _ in dataloader]
        # meta_optimizer.zero_grad()
        np.random.shuffle(time_steps)
        for step in time_steps:
            step = (step-1)
            meta_batch = dataloader[step%len(dataloader)]
            head = heads[step]
            # model.reset_weights(head)
            effective_batch_size = meta_batch['context']['x'].shape[0]
            sample = dict_to_gpu(meta_batch)
            step_loss = 0
            # backbone_clone = backbone_clones[step]
            loss = train_head(head, sample['query']['x'],sample['query']['y'], head_grad=True, backbone_grad=True, base_grad=True, train_steps=inner_steps, multiplier=(1-(outer_step+1)/outer_steps))
            # adapt_loss = loss_func(head(sample['query']['x'][0]), sample['query']['y'][0].unsqueeze(-1))
            # step_loss += adapt_loss
            # step_loss = step_loss/effective_batch_size/len(time_steps)
            # backbone.requires_grad_(True)
            # step_loss.backward()
            # meta_optimizer.step()
            # config.log({"loss": step_loss.item()})
            total_loss += loss
            if replay_meta_batch is None:
                replay_meta_batch = meta_batch
                replay_head = heads[step]
            PSNR = evaluate_one_timestep(head, split_total_coords, meta_batch, step)
            # config.log({"PSNR": PSNR})
            mean_PSNR += PSNR
        # total_loss.backward()
        # meta_optimizer.step()
        mean_PSNR = mean_PSNR/len(dataloader)
        for replay in replay_buffer:
            meta_batch = replay['meta_batch']
            head = replay['head']
            sample = dict_to_gpu(meta_batch)
            train_head(head, sample['context']['x'][0],sample['context']['y'][0], head_grad=False, backbone_grad=False, base_grad=True, train_steps=inner_steps, multiplier=1/len(replay_buffer))
            adapt_loss = loss_func(head(sample['query']['x'][0]), sample['query']['y'][0].unsqueeze(-1))
            adapt_loss.backward()
            # step_loss.backward()
            # meta_optimizer.step()
            total_loss += adapt_loss.item()
        pbar.update(1)
        pbar.set_description(f"Backbone Loss: {total_loss/len(dataloader)}, PSNR: {mean_PSNR}")
    # add to replay
    # replay_buffer.append({'meta_batch': replay_meta_batch, 'head': replay_head})


    # torch.save(backbones[0], f"/mnt/d/tmp/metaINR/backbone.pth")
    # torch.save(base, f"/mnt/d/tmp/metaINR/base.pth")
    # torch.save(heads[0], f"/mnt/d/tmp/metaINR/head.pth")


def train_head(head, sample_x, sample_y, head_grad=True, backbone_grad=False, base_grad=False, train_steps=eval_steps, multiplier=1.0, backbone=None, lr=5e-5):
    if head.backbone is None:
        head.backbone = backbones[-1]
        # backbone.heads.append(head)
    params = []
    if head_grad:
        params.append({"params": head.net.parameters(), "lr": lr})
    if backbone_grad:
        params.append({"params": head.backbone.net.parameters(), "lr": lr/5*multiplier})
    if base_grad:
        params.append({"params": base.net.parameters(), "lr": lr/25*multiplier})
    optimizer = torch.optim.Adam(params, lr=lr)
    total_loss = 0
    for i in range(train_steps):
        optimizer.zero_grad()
        preds = head(sample_x)
        loss = loss_func(preds, sample_y.unsqueeze(-1))
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        if backbone is not None:
            backbone.requires_grad_(True)
            backbone.adapt(loss)
            backbone.requires_grad_(False)
    # config.log({"loss": total_loss/train_steps})
    return total_loss/train_steps

def evaluate_one_timestep(head, split_total_coords, meta_batch, step):
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
    # saveDat(v_res, f"/mnt/d/tmp/preds{step:04d}.raw")
    return PSNR

def run():
    global base
    dataset = MetaDataset(config.target_dataset, config.target_var, config.test_timesteps,
                            dims=config.get_dims_of_dataset(config.target_dataset),
                            s=4)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for i in range(1, 81, 10):
        add_backbone()
        train_backbone(range(i, i+10, 1))

    # sample = dict_to_gpu(next(iter(dataloader)))
    # heads[0].backbone = backbones[-1]
    # optimizer = torch.optim.Adam(heads[0].parameters(), lr=1e-4)
    # step_loss = 0.0
    # for _ in tqdm(range(6), leave=False, desc="Finetuning"):
    #     step_loss = 0.0
    #     # learner = model.clone()
    #     for _ in range(500):
    #         optimizer.zero_grad()
    #         support_preds = heads[0](sample['query']['x'])
    #         step_loss = loss_func(support_preds, sample['query']['y'].unsqueeze(-1))
    #         step_loss.backward()
    #         optimizer.step()

    PSNR_list = []
    for step, meta_batch in tqdm(enumerate(dataloader), desc="Inferring", leave=False):
        if heads[step].backbone is None:
            break
        PSNR = evaluate_one_timestep(heads[step], torch.split(dataset.total_coords, 32000, dim=0), meta_batch, step)
        PSNR_list.append(PSNR)

    torch.save(base, f"/mnt/d/tmp/metaINR/base.pth")
    torch.save(backbones, f"/mnt/d/tmp/metaINR/backbone.pth")
    torch.save(heads, f"/mnt/d/tmp/metaINR/head.pth")
    print(f"Average PSNR: {np.mean(PSNR_list)}")
    print("PSNR_list: ", PSNR_list)

if __name__ == "__main__":
    run()