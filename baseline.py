"""
VP Baseline

A simple pretrain baseline for comparison with MetaINR.
Instead of using inner steps to guide training of the Meta Model, this script takes the naive approach of directly training the meta model on the dataset.
"""


from models import SIREN
from dataio import *
from collections.abc import Mapping
from tqdm import tqdm
import config
from torch import nn
import fire
from copy import deepcopy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def shuffle_and_batch(total_coords, total_values, batch_size, s=1):
    indices = torch.randperm(total_coords.shape[0]//s)
    total_coords = total_coords[indices]
    total_values = total_values[indices]

    split_coords = torch.split(total_coords, batch_size, dim=0)
    split_values = torch.split(total_values, batch_size, dim=0)
    return split_coords, split_values


# todo: [] add train from scratch code
# todo: [] find a way to measure how to determine which data can be best optmise by meta-learning

def run(dataset="vorts", var="default", train=True, ts_range=None, lr=1e-4, fast_lr=1e-4, adapt_lr=1e-5):
    if ts_range is None or len(ts_range) != 2:
        ts_range = (10, 25)
    var = str(var)
    config.target_dataset = dataset
    config.target_var = var
    config.seed_everything(42)
    config.set_status("baseline-train")
    meta_lr = float(lr)
    fast_lr = float(fast_lr)
    config.log({"lr": meta_lr, "adapt_lr": adapt_lr})
    outer_steps = 600
    inner_steps = 16
    BatchSize = 50000

    net = SIREN(in_features=3, out_features=1, init_features=64, num_res=5)  # num_res kind of sensitive
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=meta_lr)

    loss_func = nn.MSELoss()

    # train
    if train:
        total_pretrain_time = 0
        train_dataset = MetaDataset(dataset, var, t_range=ts_range, s=4, subsample_half=True)
        tic = time.time()
        for step in tqdm(range(outer_steps*inner_steps)):
            # randomly select half
            for ind in range(len(train_dataset)):
                total_coords = train_dataset[ind]["all"]["x"]
                total_values = train_dataset[ind]["all"]["y"]
                split_coords, split_values = shuffle_and_batch(total_coords, total_values, BatchSize)
                step_loss = 0
                for t_coords, t_value in zip(split_coords, split_values):
                    t_coords = t_coords.to(device)
                    t_value = t_value.to(device)
                    preds = net(t_coords).squeeze(-1)
                    loss = loss_func(preds, t_value)
                    step_loss += loss
                optimizer.zero_grad()
                step_loss.backward()
                optimizer.step()
            if step%inner_steps == 0:
                config.log({"loss": step_loss})

        total_pretrain_time += time.time()-tic
        os.makedirs(config.model_dir, exist_ok=True)
        os.makedirs(config.model_dir+f"{dataset}_{var}", exist_ok=True)
        try:
            torch.save(net.state_dict(), config.model_dir+f"{dataset}_{var}/baseline_{ts_range[0]}_{ts_range[1]}.pth")
        except Exception as e:
            print(e)
        config.log({"total_pretrain_time": total_pretrain_time})
    else:
        net.load_state_dict(torch.load(config.model_dir+f"{dataset}_{var}/baseline_{ts_range[0]}_{ts_range[1]}.pth"))

    config.set_status("baseline-eval")

    # evaluation
    total_encoding_time = 0.0
    PSNRs = []
    eval_batch = 50  # avoid memory overflow
    ts_batch_range = list(range(ts_range[0], ts_range[1], eval_batch))
    pbar = tqdm(total=ts_range[1]-ts_range[0])
    for batch_num, ts_start in enumerate(ts_batch_range):
        ts_end = min(ts_start+eval_batch, ts_range[1])
        full_dataset = MetaDataset(dataset, var, t_range=(ts_start, ts_end), s=1)
        full_dataloader = DataLoader(full_dataset, batch_size=1, shuffle=False, num_workers=0)
        total_coords = full_dataset.total_coords
        split_total_coords = torch.split(total_coords, BatchSize, dim=0)
        for inside_num, meta_batch in enumerate(full_dataloader):
            steps = batch_num*eval_batch+inside_num
            # init
            train_coords = meta_batch['all']['x'].squeeze()
            train_value = meta_batch['all']['y'].squeeze()

            # encoding
            tic = time.time()

            learner = deepcopy(net)
            optimizer = torch.optim.Adam(learner.parameters(), lr=float(adapt_lr))

            for _ in tqdm(range(inner_steps)):
                # shuffle the data
                indices = torch.randperm(train_coords.shape[0])
                train_coords = train_coords[indices]
                train_value = train_value[indices]

                split_coords = torch.split(train_coords, BatchSize, dim=0)
                split_values = torch.split(train_value, BatchSize, dim=0)
                for t_coords, t_value in zip(split_coords, split_values):
                    t_coords = t_coords.to(device)
                    t_value = t_value.to(device)
                    preds = learner(t_coords)
                    loss = loss_func(preds, t_value.unsqueeze(-1))
                    # learner.adapt(loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # config.log({"loss": loss})
            toc = time.time()
            total_encoding_time += toc-tic

            # decoding
            v_res = []
            for inf_coords in split_total_coords:
                inf_coords = inf_coords.to(device)
                preds = learner(inf_coords).detach().squeeze().cpu().numpy()
                v_res += list(preds)
            v_res = np.asarray(v_res, dtype=np.float32)
            y_vals = meta_batch['total']['y'].squeeze().cpu()
            y_vals = np.asarray(y_vals, dtype=np.float32)
            GT_range = y_vals.max()-y_vals.min()
            MSE = np.mean((v_res-y_vals)**2)
            PSNR = 20*np.log10(GT_range)-10*np.log10(MSE)
            PSNRs.append(PSNR)
            config.log({"PSNR": PSNR})
            try:
                os.makedirs(config.model_dir, exist_ok=True)
                os.makedirs(config.model_dir+f"{dataset}_{var}", exist_ok=True)
                torch.save(learner.state_dict(), config.model_dir+f"{dataset}_{var}/eval_baseline_{ts_range[0]+steps}.pth")
            except Exception as e:
                print(e)
            pbar.update(1)
            pbar.set_description(f"volume time step: {steps}, PSNR: {PSNR}")

    print("Total encoding time: ", total_encoding_time)
    config.log({"total_encoding_time": total_encoding_time})
    config.log({"average_PSNR": np.mean(PSNRs)})


if __name__ == '__main__':
    fire.Fire(run)
