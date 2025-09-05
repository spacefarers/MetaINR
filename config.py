import platform
import torch
import numpy as np
import neptune
from tqdm import tqdm
import json

# Each dataset lives in root_data_dir/{dataset_name}/
# dataset.json holds {"name","dims","vars","total_samples"}
# Each var in "vars" = subfolder containing total_samples .raw files
# Files named {name}-{var}-{index}.raw, index starts at 1
# Loader uses dataset.json to resolve dims/vars/paths

# root_data_dir/half-cylinder/
# ├── dataset.json
# ├── 160/
# │   ├── half-cylinder-160-1.raw
# │   └── ...
# ├── 320/
# │   └── half-cylinder-320-*.raw
# ├── 640/
# │   └── half-cylinder-640-*.raw
# └── 6400/
#     └── half-cylinder-6400-*.raw

# dataset.json example:
# {
#   "name": "half-cylinder",
#   "dims": [640, 240, 80],
#   "vars": ["160", "320", "640", "6400"],
#   "total_samples": 100
# }


machine = platform.node()

# these settings will be overwritten in the main program, just set some default values here feel free to ignore them
target_dataset = "vorts"
target_var = "default"

enabled_replay = False
pretraining = False
INR_training = False
transferring = False
baseline_experiment = False
device = torch.device('cuda')
batch_size = 1

root_data_dir = '/mnt/d/data/'
model_dir = '/mnt/d/models/'
results_dir = '/mnt/d/results/'

interval = 2
crop_times = 4
# low_res_size for half-cylinder: [160, 60, 20]
# low_res_size for hurricane: [125,125,25]
# low_res_size for vorts: [32,32,32]
crop_size = [16, 16, 16]  # must be multiples of 8 and smaller than low res size
scale = 4
load_ensemble_model = False
run_id = None
lr = (1e-5, 4e-5)


train_data_split = 4  # number of datapoints used for training

run_cycle = None
ensemble_iter = None
logging_init = False
enable_restorer = False

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')

seed_everything(42)
log_obj = None


def get_flag():
    if pretraining:
        flag = 'pretrain'
    elif INR_training:
        flag = 'INR'
    elif baseline_experiment:
        flag = 'baseline'
    elif transferring:
        flag = 'transfer'
    else:
        flag = 'other'
    return flag


def log(data):
    global log_obj
    for key, value in data.items():
        tqdm.write(f"{key}: {value}")
        # if you want to log to w&b do it here


tracking_obj = {}


def track(values):
    for key, value in values.items():
        if type(value) is torch.Tensor:
            value = value.mean().item()
        if key in tracking_obj:
            tracking_obj[key].append(value)
        else:
            tracking_obj[key] = [value]


def view(key):
    results = np.mean(tracking_obj[key])
    del tracking_obj[key]
    return results


def log_all(direct_log=True):
    return_result = {}
    for k in list(tracking_obj.keys()):
        if direct_log:
            log({k: view(k)})
        else:
            return_result[k] = view(k)
    return return_result



json_data = {}


def load_json_data(dataset):
    global json_data
    data_dir = root_data_dir+dataset+'/'
    dataset_json = data_dir+'dataset.json'
    json_data[dataset] = json.load(open(dataset_json))


def get_dims_of_dataset(dataset):
    global json_data
    if dataset not in json_data:
        load_json_data(dataset)
    dims = json_data[dataset]['dims']
    return dims


def get_size_of_dataset(dataset):
    global json_data
    if dataset not in json_data:
        load_json_data(dataset)
    size = json_data[dataset]['total_samples']
    return size


test_timesteps = range(1, get_size_of_dataset(target_dataset)+1)