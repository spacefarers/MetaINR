"""
MetaINR

Efficiently adapt to time series simulation data with MetaINR.
This method is based on the INR architecture and uses a meta-learning approach to adapt to new time series data.
The model is trained on a range of time steps and can be adapted to new time steps with a few gradient steps.

Achieves significantly better performance than training INR from scratch or using a simple baseline pretrained model.
"""

from models import SIREN
from dataio import *
from collections.abc import Mapping
from tqdm import tqdm
import config
from torch import nn
import fire
from copy import deepcopy
import time
import os
import numpy as np

# Set the device for PyTorch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_func = nn.MSELoss()  # Mean Squared Error loss function

def dict_to_gpu(ob):
    """Recursively move a dictionary or tensor to the GPU."""
    if isinstance(ob, Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    else:
        return ob.cuda()

def get_volumes(paths):
    """Read and concatenate volumes from given file paths."""
    volumes = []
    for path in paths:
        v = readDat(path)  # Function to read data from the path
        volumes.append(torch.tensor(v, dtype=torch.float32))  # Convert to tensor
    volumes = torch.cat(volumes, dim=0)  # Concatenate volumes along the first dimension
    return volumes

def shuffle_and_batch(total_coords, total_values, BatchSize, s=1):
    """Shuffle and batch the coordinates and values."""
    indices = torch.randperm(total_coords.shape[0] // s)  # Random permutation of indices
    total_coords = total_coords[indices]
    total_values = total_values[indices]

    # Split into batches
    split_coords = torch.split(total_coords, BatchSize, dim=0)
    split_values = torch.split(total_values, BatchSize, dim=0)
    return split_coords, split_values

def fast_adapt(batch, learner, adapt_opt, adaptation_steps, batch_size):
    """Perform fast adaptation for the learner on the given batch."""
    data, labels = batch
    total_loss = 0
    for step in range(adaptation_steps):
        split_coords, split_values = shuffle_and_batch(data, labels, batch_size)  # Shuffle and batch data
        for t_coords, t_value in zip(split_coords, split_values):
            t_coords = t_coords.to(device)
            t_value = t_value.to(device)
            preds = learner(t_coords).squeeze(-1)  # Get predictions
            loss = loss_func(preds, t_value)  # Calculate loss
            adapt_opt.zero_grad()
            loss.backward()  # Backpropagation
            total_loss += loss
            adapt_opt.step()  # Update parameters
    return total_loss

def evaluate(batch, learner, batch_size):
    """Evaluate the learner on the given batch and return PSNR."""
    data, labels = batch
    total_loss = 0
    split_coords, split_values = shuffle_and_batch(data, labels, batch_size)
    v_res = []
    y_vals = []

    for t_coords, t_value in zip(split_coords, split_values):
        t_coords = t_coords.to(device)
        t_value = t_value.to(device)
        preds = learner(t_coords).squeeze(-1)  # Get predictions
        v_res += list(preds.cpu().detach().numpy())  # Store predictions
        y_vals += list(t_value.cpu().detach().numpy())  # Store true values

    v_res = np.asarray(v_res, dtype=np.float32)
    y_vals = np.asarray(y_vals, dtype=np.float32)

    # Calculate PSNR
    GT_range = y_vals.max() - y_vals.min()
    MSE = np.mean((v_res - y_vals) ** 2)
    PSNR = 20 * np.log10(GT_range) - 10 * np.log10(MSE)
    return PSNR

def run(dataset="half-cylinder", var="640", train=True, ts_range=None, lr=1e-4, fast_lr=1e-4, adapt_lr=1e-5):
    """Main function to run the MetaINR training or evaluation."""
    # change ts_range to limit the range of time steps to run on
    if ts_range is None or len(ts_range) != 2:
        ts_range = (0, 598)  # Default time range
    var = str(var)
    config.target_dataset = dataset
    config.target_var = var
    config.seed_everything(42)  # Set random seed for reproducibility

    # Learning rates
    meta_lr = float(lr)
    fast_lr = float(fast_lr)
    config.log({"lr": meta_lr, "fast_lr": fast_lr, "adapt_lr": adapt_lr})

    # Define training parameters
    outer_steps = 500
    inner_steps = 16
    BatchSize = 50000

    # Initialize the SIREN model
    net = SIREN(in_features=3, out_features=1, init_features=64, num_res=5)
    net = net.to(device)  # Move model to GPU
    opt = torch.optim.Adam(net.parameters(), lr=meta_lr)  # Outer optimizer
    adapt_opt = torch.optim.Adam(net.parameters(), lr=fast_lr)  # Inner optimizer
    adapt_opt_state = adapt_opt.state_dict()  # Save optimizer state

    # Training phase
    if train:
        train_dataset = MetaDataset(dataset, var, t_range=ts_range, s=4, subsample_half=True)
        total_pretrain_time = 0
        tic = time.time()

        for step in tqdm(range(outer_steps)):
            opt.zero_grad()  # Zero gradients for outer optimizer

            # Randomly select half the time steps
            for p in net.parameters():
                p.grad = torch.zeros_like(p.data)  # Reset gradients

            total_loss = 0
            for ind in range(len(train_dataset)):
                learner = deepcopy(net)  # Create a copy of the model for adaptation
                adapt_opt = torch.optim.Adam(learner.parameters(), lr=fast_lr)  # New optimizer for learner
                adapt_opt.load_state_dict(adapt_opt_state)  # Load previous optimizer state
                train_coords = train_dataset[ind]["all"]["x"]  # Get training coordinates
                train_value = train_dataset[ind]["all"]["y"]  # Get training values
                batch = (train_coords, train_value)  # Create batch
                loss = fast_adapt(batch, learner, adapt_opt, inner_steps, BatchSize)  # Adapt learner
                total_loss += loss  # Accumulate loss

                # Update the outer model
                adapt_opt_state = adapt_opt.state_dict()
                for p, l in zip(net.parameters(), learner.parameters()):
                    p.grad.data.add_(l.data, alpha=-1.0)  # Update gradients

            # Average gradients and update outer model
            for p in net.parameters():
                p.grad.data.mul_(1.0 / (len(train_dataset))).add_(p.data)
            opt.step()  # Step the outer optimizer

            config.log({"loss": total_loss})  # Log loss

        total_pretrain_time += time.time() - tic
        os.makedirs(config.model_dir, exist_ok=True)  # Create model directory if not exists
        os.makedirs(config.model_dir + f"{dataset}_{var}", exist_ok=True)

        # Save the model state
        try:
            torch.save(net.state_dict(), config.model_dir + f"{dataset}_{var}/{ts_range[0]}_{ts_range[1]}.pth")
        except Exception as e:
            print(e)

        config.log({"total_pretrain_time": total_pretrain_time})  # Log pretraining time
    else:
        # Load the pre-trained model
        net.load_state_dict(torch.load(config.model_dir + f"{dataset}_{var}/{ts_range[0]}_{ts_range[1]}.pth"))

    # Evaluation phase
    total_encoding_time = 0.0
    PSNRs = []
    eval_batch = 50  # Avoid memory overflow
    ts_batch_range = list(range(ts_range[0], ts_range[1], eval_batch))
    pbar = tqdm(total=ts_range[1] - ts_range[0])  # Progress bar for evaluation

    for batch_num, ts_start in enumerate(ts_batch_range):
        ts_end = min(ts_start + eval_batch, ts_range[1])
        full_dataset = MetaDataset(dataset, var, t_range=(ts_start, ts_end), s=1)
        full_dataloader = DataLoader(full_dataset, batch_size=1, shuffle=False, num_workers=0)
        total_coords = full_dataset.total_coords
        split_total_coords = torch.split(total_coords, BatchSize, dim=0)

        for inside_num, meta_batch in enumerate(full_dataloader):
            steps = batch_num * eval_batch + inside_num

            # Initialize training coordinates and values
            train_coords = meta_batch['all']['x'].squeeze()
            train_value = meta_batch['all']['y'].squeeze()

            # Encoding phase
            tic = time.time()
            learner = deepcopy(net)  # Create a copy of the model for evaluation
            optimizer = torch.optim.Adam(learner.parameters(), lr=float(adapt_lr))  # Optimizer for learner

            for step in tqdm(range(inner_steps)):
                # Shuffle the data
                indices = torch.randperm(train_coords.shape[0])
                train_coords = train_coords[indices]
                train_value = train_value[indices]

                split_coords = torch.split(train_coords, BatchSize, dim=0)
                split_values = torch.split(train_value, BatchSize, dim=0)

                for t_coords, t_value in zip(split_coords, split_values):
                    t_coords = t_coords.to(device)
                    t_value = t_value.to(device)
                    preds = learner(t_coords)  # Get predictions
                    loss = loss_func(preds, t_value.unsqueeze(-1))  # Calculate loss
                    optimizer.zero_grad()
                    loss.backward()  # Backpropagation
                    optimizer.step()  # Update learner parameters

            toc = time.time()
            total_encoding_time += toc - tic  # Accumulate encoding time

            # Decoding phase
            v_res = []
            for inf_coords in split_total_coords:
                inf_coords = inf_coords.to(device)
                preds = learner(inf_coords).detach().squeeze().cpu().numpy()  # Get predictions
                v_res += list(preds)  # Store predictions

            v_res = np.asarray(v_res, dtype=np.float32)
            y_vals = meta_batch['total']['y'].squeeze().cpu()
            y_vals = np.asarray(y_vals, dtype=np.float32)

            # Calculate PSNR
            GT_range = y_vals.max() - y_vals.min()
            MSE = np.mean((v_res - y_vals) ** 2)
            PSNR = 20 * np.log10(GT_range) - 10 * np.log10(MSE)
            PSNRs.append(PSNR)  # Store PSNR
            config.log({"PSNR": PSNR})  # Log PSNR

            # Save the learner state
            try:
                os.makedirs(config.model_dir, exist_ok=True)
                os.makedirs(config.model_dir + f"{dataset}_{var}", exist_ok=True)
                torch.save(learner.state_dict(), config.model_dir + f"{dataset}_{var}/eval_metainr_{ts_range[0] + steps}.pth")
            except Exception as e:
                print(e)

            pbar.update(1)  # Update progress bar
            pbar.set_description(f"volume time step: {steps}, PSNR: {PSNR}")

    print("Total encoding time: ", total_encoding_time)  # Print total encoding time
    config.log({"total_encoding_time": total_encoding_time})  # Log encoding time
    config.log({"average_PSNR": np.mean(PSNRs)})  # Log average PSNR

if __name__ == '__main__':
    fire.Fire(run)  # Run the main function with fire
