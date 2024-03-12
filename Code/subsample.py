from utils import *
from tools import * 
from tqdm import tqdm

if __name__ == "__main__":
    v_dir_path = "/home/dullpigeon/Desktop/MetaINR/Code/Data/SubSample"
    dims = [640,240,80]
    paths = getFilePathsInDir(v_dir_path)
    subsample_indices = Space_Subsample(dims,s=4)
    
    for path in tqdm(paths):
        volume = readDat(path)
        volume = volume[subsample_indices]
        saveDat(volume,path)
        