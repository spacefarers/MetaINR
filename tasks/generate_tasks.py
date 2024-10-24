# generates a csv of tasks, each task is intended to be run on one machine

rules = {
    ("vorts", "default"): [10, 25],
    ("tangaroa", "default"): [10, 25],
    ("ionization", "GT"): [70, 100],
    ("combustion", "YOH"): [70, 85],
    ("earthquake", "default"): [150, 250],
    ("half-cylinder", "640"): [10, 25],
    ("supernova", "default"): [10, 25],
    ("fivejet", "default"): [10, 25],
}

targets = [
    # "baseline",
    "INR",
    # "MetaINR",
]

# write the command to a csv
with open("tasks.csv", "w") as f:
    f.write("target,dataset,var,ts_range_st,ts_range_end\n")
    for target in targets:
        for (dataset, var), ts_range in rules.items():
            f.write(f"{target},{dataset},{var},{ts_range[0]},{ts_range[1]}\n")
