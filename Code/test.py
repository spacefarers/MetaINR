from main import evaluate_one_timestep
import torch

columns = torch.load("/mnt/d/tmp/metaINR/columns.pth")
linears = torch.load("/mnt/d/tmp/metaINR/linears.pth")

# verify results