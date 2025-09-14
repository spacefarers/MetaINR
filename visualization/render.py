from tqdm import tqdm

import config
from dataio import *
from models import SIREN

rules = {
    ("vorts", "default"): [[10, 25]],
    ("tangaroa", "default"): [[120, 150]],
    ("ionization", "GT"): [[70, 100]],
    ("half-cylinder", "640"): [[80, 100]],
}
targets = [
    "MetaINR",
    "Baseline",
    "INR100",
    "INR16",
    "GT",
]
BatchSize = 50000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for (dataset, var), ts_ranges in rules.items():
    for ts_range in ts_ranges:
        for target in targets:
            if target == "MetaINR":
                load_suffix = "metainr"
            elif target == "Baseline":
                load_suffix = "baseline"
            elif target == "INR100":
                load_suffix = "inr100"
            elif target == "INR16":
                load_suffix = "inr16"
            elif target == "GT":
                load_suffix = "gt"
            else:
                raise ValueError(f"Unknown target: {target}")
            print(f"Running {target} on {dataset} {var} {ts_range}")

            if target == "GT":
                # simply copy over the ground truth
                meta_dataset = MetaDataset(dataset, var, t_range=ts_range, s=1)
                for ind, t in enumerate(tqdm(range(ts_range[0], ts_range[1]))):
                    y_vals = meta_dataset[ind]["all"]["y"].squeeze().cpu().detach().numpy()
                    y_vals = np.asarray(y_vals, dtype=np.float32)
                    os.makedirs(
                        config.results_dir + f"{dataset}_{var}/{load_suffix}/",
                        exist_ok=True,
                    )
                    saveDat(
                        y_vals,
                        config.results_dir + f"{dataset}_{var}/{load_suffix}/preds_{load_suffix}_{t}.raw",
                    )
                continue

            dims = config.get_dims_of_dataset(dataset)
            print(dims)
            total_coords = torch.tensor(get_mgrid(dims, dim=3, s=1), dtype=torch.float32)
            split_total_coords = torch.split(total_coords, BatchSize, dim=0)

            net = SIREN(in_features=3, out_features=1, init_features=64, num_res=5)
            for t in tqdm(range(ts_range[0], ts_range[1])):
                net.load_state_dict(torch.load(config.model_dir + f"{dataset}_{var}/eval_{load_suffix}_{t}.pth"))
                net = net.to(device)
                net.eval()
                # decoding
                v_res = []
                for inf_coords in split_total_coords:
                    inf_coords = inf_coords.to(device)
                    preds = net(inf_coords).detach().squeeze().cpu().numpy()
                    v_res += list(preds)
                v_res = np.asarray(v_res, dtype=np.float32)
                # normalize to 0 to 1
                # v_res = (v_res-v_res.min())/(v_res.max()-v_res.min())
                os.makedirs(
                    config.results_dir + f"{dataset}_{var}/{load_suffix}/",
                    exist_ok=True,
                )
                saveDat(
                    v_res,
                    config.results_dir + f"{dataset}_{var}/{load_suffix}/preds_{load_suffix}_{t}.raw",
                )
