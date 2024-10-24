"""
INR as a baseline for comparison with MetaINR.

Runs INR of a 5-layer SIREN model on the given dataset and variable.
Computes Average PSNR for the given time steps range.
"""


from models import CoordNet, SIREN
from dataio import *
from collections.abc import Mapping
from tqdm import tqdm
from torch import nn
import fire

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


def run(dataset="vorts", var="default", ts_range=None):
    lr = 1e-5
    config.log({"lr": lr})
    train_iterations = 100
    BatchSize = 50000

    loss_func = nn.MSELoss()
    if ts_range is None or len(ts_range) != 2:
        ts_range = (10, 25)

    test_dataset = MetaDataset(dataset, var, t_range=ts_range, s=1)

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    config.set_status("INR encoding")


    # evaluation
    total_encoding_time = 0.0
    pbar = tqdm(total=len(test_dataset))
    total_coords = test_dataset.total_coords
    split_total_coords = torch.split(total_coords, BatchSize, dim=0)
    PSNRs = []
    for steps, meta_batch in enumerate(test_dataloader):
        # init
        train_coords = meta_batch['all']['x'].squeeze()
        train_value = meta_batch['all']['y'].squeeze()

        model = SIREN(in_features=3, out_features=1, init_features=64, num_res=5).to(device)  # num_res kind of sensitive
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # encoding
        tic = time.time()
        for st in tqdm(range(train_iterations)):
            # shuffle the data
            indices = torch.randperm(train_coords.shape[0])
            train_coords = train_coords[indices]
            train_value = train_value[indices]

            split_coords = torch.split(train_coords, BatchSize, dim=0)
            split_values = torch.split(train_value, BatchSize, dim=0)
            for t_coords, t_value in zip(split_coords, split_values):
                t_coords = t_coords.to(device)
                t_value = t_value.to(device)
                preds = model(t_coords)
                loss = loss_func(preds, t_value.unsqueeze(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if st % 50 == 0:
                config.log({"loss": loss})
        toc = time.time()
        total_encoding_time += toc-tic

        # decoding
        v_res = []
        for inf_coords in split_total_coords:
            inf_coords = inf_coords.to(device)
            preds = model(inf_coords).detach().squeeze().cpu().numpy()
            v_res += list(preds)
        v_res = np.asarray(v_res, dtype=np.float32)
        y_vals = meta_batch['total']['y'].squeeze().cpu()
        y_vals = np.asarray(y_vals, dtype=np.float32)
        GT_range = y_vals.max()-y_vals.min()
        MSE = np.mean((v_res-y_vals)**2)
        PSNR = 20*np.log10(GT_range)-10*np.log10(MSE)
        PSNRs.append(PSNR)
        config.log({"PSNR": PSNR})
        pbar.update(1)
        pbar.set_description(f"volume time step: {steps}, PSNR: {PSNR}")

    print("Total encoding time: ", total_encoding_time)
    config.log({"total_encoding_time": total_encoding_time})
    config.log({"average_PSNR": np.mean(PSNRs)})




if __name__ == "__main__":
    fire.Fire(run)
