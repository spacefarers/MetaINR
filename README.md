# **Meta-INR: Efficient Encoding of Volumetric Data via Meta-Learning**

**Paper (IEEE Xplore):** https://ieeexplore.ieee.org/abstract/document/11021021

Meta-INR is a two-stage pretraining + fast-adaptation pipeline for implicit neural representations (INRs) of volumetric data. It learns a strong, generalizable initialization from a small spatiotemporal subsample of a dataset, then adapts to each volume with a few gradient steps. This yields faster convergence and competitive or better fidelity versus training from scratch.

## **What You Can Do**
- Meta-pretrain a SIREN-based INR on a sparse subset of a dataset.
- Rapidly adapt to each timestep/ensemble volume and save per-volume INRs.
- Run baselines: training SIREN from scratch and a naïve pretrained SIREN.
- Decode saved models into reconstructed volumes for rendering and evaluation.

## **Repository Structure**
- `main.py` — Meta-INR meta-pretraining and per-volume adaptation/evaluation.
- `baseline.py` — Naïve pretraining baseline (no inner-loop) + adaptation.
- `INR_encoding.py` — Train an INR from scratch per volume (SIREN baseline).
- `dataio.py` — Dataset loaders for meta-pretraining/adaptation.
- `models.py` — SIREN and related layers/blocks.
- `config.py` — Paths, dataset metadata, seeds, and logging helpers.
- `utils/` — I/O and file utilities used across the codebase.
- `visualization/render.py` — Decode checkpoints to raw volumes for rendering.
- `tasks/` — CSV-driven batch runs for clusters (optional).


## **Requirements**
- Python 3.8+
- PyTorch (CUDA recommended)
- Python packages: `numpy`, `tqdm`, `fire`, `einops`, `icecream`, `Pillow`
  - Quick install: `pip install torch numpy tqdm fire einops icecream pillow`
  - Optional: `neptune` (only for the analysis utilities in `visualization/make_table.py`)

## **Configure Paths**
Edit `config.py` to match your machine:
- `root_data_dir` — where datasets live (expects subfolders per dataset).
- `model_dir` — where model checkpoints are saved.
- `results_dir` — where decoded volumes are written.

Defaults point to `/mnt/d/...`; update to your local folders.

## **Dataset Setup**
Datasets are organized as one folder per dataset name under `root_data_dir`, with a `dataset.json` and one subfolder per variable (e.g., a single `default` variable or named variables like `YOH`). Each variable folder contains volume files (`.raw`, float32, little-endian), one per timestep/ensemble sample.

**Directory layout (example):**
```
{root_data_dir}/half-cylinder/
  ├── dataset.json
  ├── 160/                  # a variable name
  │   ├── half-cylinder-160-1.raw
  │   ├── half-cylinder-160-2.raw
  │   └── ...
  ├── 320/
  ├── 640/
  └── 6400/
```

**Required `dataset.json` (example):**
```json
{
  "name": "half-cylinder",
  "dims": [640, 240, 80],             
  "vars": ["160", "320", "640", "6400"],
  "total_samples": 100                 
}
```
  - `dims` are `[X, Y, Z]` matching the raw file’s voxel order when flattened.
  - `vars` lists the available variable subfolders under the dataset.
  - `total_samples` is the number of volumes per variable.

- File naming convention:
  - `{name}-{var}-{index}.raw`, with `index` starting at 1 and increasing by 1.
  - The loader sorts files by the trailing hyphen-separated integer.

- Data format:
  - Each `.raw` is a flat float32 (little-endian) array of length `X*Y*Z`.
  - Values are normalized internally as needed during training.

If your dataset has a single variable, use a single subfolder such as `default` and set `vars: ["default"]`. For multi-variable datasets (e.g., `combustion/YOH`), use that variable name as the subfolder.


**Sample dataset (for quick start):**
- Google Drive: https://drive.google.com/drive/folders/1C-03Mk6kkCWwOaGUN4X2VF1Gu8hIHdSr?usp=sharing
- Download and place the dataset folder directly under your `root_data_dir`, preserving its internal structure and `dataset.json`.

## **Running the Models**

### **1. Meta-INR (main.py)**
Meta-INR runs two phases in one command by default: meta-pretraining on a sparse subsample (inner/outer loops), then per-volume fast adaptation across the requested timesteps.

**Examples:**
```bash
# Meta-INR on half-cylinder at var=640, timesteps [80,100)
python main.py --dataset=half-cylinder --var=640 --ts_range="(80,100)" \
               --lr=1e-4 --fast_lr=1e-4 --adapt_lr=1e-5

# Meta-INR on earthquake, full range [0,598)
python main.py --dataset=earthquake --var=default --ts_range="(0,598)" \
               --lr=1e-4 --fast_lr=1e-4 --adapt_lr=1e-5
```

**Notes:**
- Default hyperparameters follow the paper: 500 outer steps, 16 inner steps, batch size 50,000, spatial subsampling `s=4` during meta-pretraining.
- Checkpoints are saved under `{model_dir}/{dataset}_{var}/`:
  - Meta-model snapshot: `{ts_start}_{ts_end}.pth`
  - Per-volume adapted models: `eval_metainr_{t}.pth` for each timestep `t`
- The console logs average PSNR across the requested range and total encoding time.

### **2. SIREN Baseline (INR_encoding.py)**
Train an INR from scratch for each volume independently using SIREN.

**Example:**
```bash
python INR_encoding.py --dataset=ionization --var=GT --ts_range="(70,100)" \
                       --train_iterations=100 --lr=1e-5
```
- Produces `eval_inr{iters}_{t}.pth` checkpoints per timestep.

### **3. Pretrained SIREN Baseline (baseline.py)**
Naïve pretrained SIREN (no inner-loop meta-learning).

**Example:**
```bash
python baseline.py --dataset=tangaroa --var=default --ts_range="(120,150)" \
                   --lr=1e-4 --adapt_lr=1e-5
```
- Produces `baseline_{ts_start}_{ts_end}.pth` (pretrained init) and `eval_baseline_{t}.pth` per timestep.

## **Recommended Ranges (Examples From The Paper)**
- `vorts/default`: `ts_range=(10,25)`
- `tangaroa/default`: `ts_range=(120,150)`
- `ionization/GT`: `ts_range=(70,100)`
- `half-cylinder/640`: `ts_range=(80,100)`
- `earthquake/default`: `ts_range=(0,598)`

## **Troubleshooting**
- **Paths:** Update `root_data_dir`, `model_dir`, and `results_dir` in `config.py` to valid locations on your machine
- **File names:** Ensure the `{name}-{var}-{index}.raw` pattern and that indexing starts at 1; the loader sorts by the trailing integer
- **Shapes:** `dataset.json` `dims` must match the flattened volume length (`X*Y*Z`)
- **Memory:** Reduce `BatchSize` in the scripts if you hit GPU memory limits
- **CPU-only:** Training works on CPU but is much slower; ensure PyTorch CPU build is installed

## **Citing**
If you use Meta-INR in your research, please cite the paper:
- Meta-INR: Efficient Encoding of Volumetric Data via Meta-Learning (IEEE PacificVis 2025)
- Link: https://ieeexplore.ieee.org/abstract/document/11021021
`