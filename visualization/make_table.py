from neptune import Project

import config
import keys
import pandas as pd
import math

project = Project(project="VRNET/MetaINR", api_token=keys.NEPTUNE_API_TOKEN, mode="read-only")
exps = project.fetch_runs_table(columns=["sys/creation_time", "sys/running_time", "sys/description", "average_PSNR", "dataset", "target"]).to_pandas().sort_values(by="sys/creation_time", ascending=False)

# no spaces allowed!
notes = [
    ["MetaINR(new) 1e-4 1e-4 1e-5", 0],
    ["VP 1e-4 1e-4 1e-5", 0],
    ["INR100", 0],
    ["INR16", 0]
]

datasets = [
    "vorts",
    "half-cylinder",
    "earthquake",
    "combustion",
    "fivejet",
    "ionization",
    "supernova",
    "tangaroa",
]

# create a table of dataset on columns and target on rows
table = {}
activated_context = None
for index, row in exps.iterrows():
    note = row["sys/description"]
    dataset = row["dataset"]
    match_prefix = [note.startswith(n) and skips == 0 for n, skips in notes]
    if any(match_prefix) and dataset in datasets:
        key = match_prefix.index(True)
        activated_context = notes.pop(key)[0]
        table[activated_context] = {}
    elif note != "":
        activated_context = None
    for i in range(len(notes)):
        if note.startswith(notes[i][0]) and notes[i][1] > 0:
            notes[i][1] -= 1
    if activated_context is not None:
        runtime_s = row["sys/running_time"]
        runtime_h = int(runtime_s // 3600)
        runtime_m = int((runtime_s % 3600) // 60)
        runtime_s = int(runtime_s % 60)
        # if no hour, display min and sec, otherwise display hour and min
        time_display = f"{runtime_h}h {runtime_m}m" if runtime_h > 0 else f"{runtime_m}m {runtime_s}s" if not math.isnan(row["average_PSNR"]) else "..."
        table[activated_context][dataset] = f"{row['average_PSNR']:.2f} ({time_display})"

table["INR100"]["fivejet"] = "43.60 (31m 19s)"
table["INR100"]["earthquake"] = "52.82 (11h 31m)"


# export to xlsx
df = pd.DataFrame(table)
df.to_excel("table.xlsx")