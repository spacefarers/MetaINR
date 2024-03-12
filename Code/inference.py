import config
import time
import torch
from tqdm import tqdm
from pathlib import Path
import json
from matplotlib import pyplot as plt
import numpy as np
from neptune.types import File



def save_plot(PSNR, PSNR_list, save_path=None, ensemble_iter=None, run_cycle=None):
    if ensemble_iter is not None:
        desc = f'#{config.run_id}{f" E.{ensemble_iter}"}: {config.target_dataset} {config.target_var} PSNR'
        name = f'PSNR-E.{ensemble_iter}'
    elif run_cycle is not None:
        desc = f'#{config.run_id}{f" C.{run_cycle}"}: {config.target_dataset} {config.target_var} PSNR'
        name = f'PSNR-C.{run_cycle}'
    else:
        desc = f'#{config.run_id}: {config.target_dataset} {config.target_var} PSNR'
        name = 'PSNR'
    plt.clf()
    axes = plt.gca()
    axes.set_ylim([0, np.max(PSNR_list) + 5])
    plt.plot(PSNR_list)
    plt.axhline(y=PSNR, color='r', linestyle='--')
    plt.xlabel('Frame')
    plt.ylabel('PSNR')
    plt.title(desc)
    plt.yticks(list(plt.yticks()[0]) + [PSNR])
    if save_path is None:
        save_path = "/mnt/d/tmp"
    plt.savefig(save_path + f'/{name}.png', dpi=300)
    config.log({"PSNR Plot": File(save_path + f'/{name}.png')})
