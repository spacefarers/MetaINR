import learn2learn as l2l
from utils import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from CoordNet import CoordNet, SIREN, Column, Linear
from tools import *
from dataio import *
from collections.abc import Mapping
import config
from inference import save_plot
from torch import nn
from termcolor import colored
from copy import deepcopy

# some times you only want small parts of volumes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dict_to_gpu(ob):
    if isinstance(ob, Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    else:
        return ob.cuda()


def get_volumes(paths):
    volumes = []
    for path in paths:
        v = readDat(path)
        volumes.append(torch.tensor(v, dtype=torch.float32))
    volumes = torch.cat(volumes, dim=0)
    return volumes

def evaluate_one_timestep(split_total_coords, linear, meta_batch, step):
    v_res = []
    for inf_coords in split_total_coords:
        inf_coords = inf_coords.to(device)
        preds = linear(inf_coords).detach().squeeze().cpu().numpy()
        v_res += list(preds)
    v_res = np.asarray(v_res, dtype=np.float32)
    y_vals = meta_batch['total']['y'].squeeze().numpy()
    GT_range = y_vals.max()-y_vals.min()
    MSE = np.mean((v_res-y_vals)**2)
    PSNR = 20*np.log10(GT_range)-10*np.log10(MSE)
    saveDat(v_res, f"/mnt/d/tmp/preds{step:04d}.raw")
    return PSNR

cycle = 0
def evaluate_all_timesteps(test_dataloader, learner):
    global cycle
    total_coords = test_dataset.total_coords
    split_total_coords = torch.split(total_coords, 32000, dim=0)
    PSNR_list = []
    for step, meta_batch in tqdm(enumerate(test_dataloader), desc="Inferring", leave=False):
        # sample = dict_to_gpu(meta_batch)
        # for i in range(eval_steps):
        #     preds = learner(sample['all']['x'])
        #     loss = loss_func(preds, sample['all']['y'].unsqueeze(-1))
        #     learner.adapt(loss)
        # inference
        v_res = []
        for inf_coords in split_total_coords:
            inf_coords = inf_coords.to(device)
            preds = learner(inf_coords).detach().squeeze().cpu().numpy()
            v_res += list(preds)
        v_res = np.asarray(v_res, dtype=np.float32)
        y_vals = meta_batch['total']['y'].squeeze().numpy()
        GT_range = y_vals.max()-y_vals.min()
        MSE = np.mean((v_res-y_vals)**2)
        PSNR = 20*np.log10(GT_range)-10*np.log10(MSE)
        PSNR_list.append(PSNR)
        # saveDat(v_res, f"./results/preds{steps:04d}.raw")
    print("PSNR: ", np.mean(PSNR_list))
    print("PSNR list: ", PSNR_list)
    save_plot(np.mean(PSNR_list), PSNR_list,run_cycle=cycle)
    cycle += 1



# todo: [] add train from scratch code
# todo: [] find a way to measure how to determine which data can be best optmise by meta-learning

if __name__ == "__main__":
    meta_lr = 5e-5  # very sensitive
    fast_lr = 0.001
    outer_steps = 50
    inner_steps = 16
    eval_steps = 128
    BatchSize = 16
    config.run_id = 100

    net = SIREN(in_features=3, out_features=1, init_features=64, num_res=3)  # num_res kind of sensitive
    model = l2l.algorithms.MAML(net, lr=fast_lr, first_order=True).to(device)

    meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
    loss_func = nn.MSELoss()

    train_dataset = MetaDataset(config.target_dataset, config.target_var, config.train_timesteps,
                                dims=config.get_dims_of_dataset(config.target_dataset),
                                s=4)
    test_dataset = MetaDataset(config.target_dataset, config.target_var, config.test_timesteps,
                               dims=config.get_dims_of_dataset(config.target_dataset),
                               s=4,
                               split='test')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # train
    iterations = 0
    # load model
    # if os.path.exists(f"./results/model.pth"):
    if os.path.exists(f"./head.pth") or False:
        # model = torch.load(f"./head.pth")
        print("model loaded")
    else:
        config.log({"stage": "Meta Training"})
        for epoch in tqdm(range(outer_steps), desc="Epoch", leave=False):
            step_loss = 0.0
            for steps, meta_batch in tqdm(enumerate(train_dataloader), desc="Batch", leave=False):
                effective_batch_size = meta_batch['context']['x'].shape[0]
                sample = dict_to_gpu(meta_batch)
                meta_optimizer.zero_grad()
                step_loss = 0.0
                for i in range(effective_batch_size):
                    learner = model.clone()
                    for _ in range(inner_steps):
                        support_preds = learner(sample['context']['x'][i])
                        support_loss = loss_func(support_preds, sample['context']['y'][i].unsqueeze(-1))
                        learner.adapt(support_loss)
                    adapt_loss = loss_func(learner(sample['query']['x'][i]), sample['query']['y'][i].unsqueeze(-1))
                    step_loss += adapt_loss
                step_loss = step_loss/effective_batch_size

                step_loss.backward()
                meta_optimizer.step()
            config.log({"loss": step_loss})
        # save model
        torch.save(model, f"./head.pth")
    config.log({"stage": "Evaluation"})
    # evaluation

    # evaluation
    pbar = tqdm(total=len(test_dataset))
    total_coords = test_dataset.total_coords
    split_total_coords = torch.split(total_coords, 32000, dim=0)
    PSNR_list = []
    for steps,meta_batch in enumerate(test_dataloader):
        sample = dict_to_gpu(meta_batch)
        learner = deepcopy(model)
        optimizer = torch.optim.Adam(learner.parameters(), lr=1e-4)
        for i in range(eval_steps):
            preds = learner(sample['all']['x'])
            loss = loss_func(preds, sample['all']['y'].unsqueeze(-1))
            # learner.adapt(loss)
            loss.backward()
            optimizer.step()
        # inference
        v_res = []
        for inf_coords in split_total_coords:
            inf_coords = inf_coords.to(device)
            preds = learner(inf_coords).detach().squeeze().cpu().numpy()
            v_res += list(preds)
        v_res = np.asarray(v_res, dtype=np.float32)
        # saveDat(v_res, f"./results/preds{steps:04d}.raw")
        v_res = np.asarray(v_res, dtype=np.float32)
        y_vals = meta_batch['total']['y'].squeeze().numpy()
        GT_range = y_vals.max()-y_vals.min()
        MSE = np.mean((v_res-y_vals)**2)
        PSNR = 20*np.log10(GT_range)-10*np.log10(MSE)
        config.log({"PSNR": PSNR})
        PSNR_list.append(PSNR)
        pbar.update(1)
        pbar.set_description(f"volume time step: {steps}")

    print("PSNR: ", np.mean(PSNR_list))
    print("PSNR list: ", PSNR_list)
    save_plot(np.mean(PSNR_list), PSNR_list)

# 44.1MB Vorts 1-80 35.68486964702606