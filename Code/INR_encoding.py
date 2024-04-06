import learn2learn as l2l
from utils import *
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from CoordNet import CoordNet, SIREN
from tools import *
from dataio import *
from collections.abc import Mapping
from torch import nn
import config

# some times you only want small parts of volumes

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
        volumes.append(torch.tensor(v,dtype=torch.float32))
    volumes = torch.cat(volumes, dim=0)
    return volumes

if __name__ == "__main__":
    config.run_id = 1
    lr  = 1e-4 
    train_iterations = 750
    BatchSize = 1
    
    
    loss_func = nn.MSELoss()
    
    test_dataset = MetaDataset(config.target_dataset,config.target_var, config.test_timesteps,
                          dims=config.get_dims_of_dataset(config.target_dataset),
                          s=4,
                          split='test')

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    
    # evaluation
    total_encoding_time = 0.0
    pbar = tqdm(total=len(test_dataset))
    total_coords = test_dataset.total_coords
    split_total_coords = torch.split(total_coords, 32000, dim=0)
    PSNR_list = []
    models = []
    for steps,meta_batch in enumerate(test_dataloader):
        # init 
        train_coords = meta_batch['all']['x']
        train_value = meta_batch['all']['y']
        dataset = torch.utils.data.TensorDataset(train_coords,train_value)
        data_loader = DataLoader(dataset, batch_size=BatchSize, shuffle=True, num_workers=0)
        model = SIREN(in_features=3, out_features=1, init_features=64,num_res=3).to(device) # num_res kind of sensitive
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # encoding 
        tic = time.time()
        iter_count = 0
        loss = 0
        while iter_count < train_iterations:
            for t_coords,t_value in data_loader:
                t_coords = t_coords.to(device)
                t_value = t_value.to(device)
                preds = model(t_coords)
                loss = loss_func(preds, t_value.unsqueeze(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter_count += 1
                if iter_count >= train_iterations:
                    break
        config.log({"loss": loss})
        toc = time.time()
        total_encoding_time += toc - tic
        models.append(model)
        
        # decoding
        v_res = []
        for inf_coords in split_total_coords:
            inf_coords = inf_coords.to(device)
            preds = model(inf_coords).detach().squeeze().cpu().numpy()
            v_res += list(preds)
        y_vals = meta_batch['total']['y'].squeeze().numpy()
        GT_range = y_vals.max() - y_vals.min()
        MSE = np.mean((y_vals - v_res) ** 2)
        PSNR = 20 * np.log10(GT_range) - 10 * np.log10(MSE)
        config.log({"PSNR": PSNR})
        PSNR_list.append(PSNR)
        # v_res = np.asarray(v_res, dtype=np.float32)
        # saveDat(v_res, f"./results/preds{steps:04d}.raw")
        pbar.update(1)
        pbar.set_description(f"volume time step: {steps}")
    print("PSNR: ", np.mean(PSNR_list))
    print("PSNR list: ", PSNR_list)
    torch.save(models, f"{config.temp_dir}/models.pth")

print("Total encoding time: ", total_encoding_time)
# 73.5MB Vorts 1-80 36.08