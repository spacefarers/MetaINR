# generates a csv of tasks, each task is intended to be run on one machine

rules = {
    # ("vorts", "default"): [[10, 25]],
    # ("tangaroa", "default"): [[120, 150]],
    # ("ionization", "GT"): [[70, 100]],
    # # ("combustion", "YOH"): [[70, 85]],
    ("earthquake", "default"): [[0, 598]],
    # ("half-cylinder", "640"): [[80, 100]],
    # # ("supernova", "default"): [[10, 25]],
    # # ("fivejet", "default"): [[10, 25]],
}

# for ts_start in range(0, 598, 30):
#     ts_end = ts_start + 30 if ts_start + 30 < 598 else 598
#     if ("earthquake", "default") not in rules:
#         rules[("earthquake", "default")] = []
#     rules[("earthquake", "default")].append([ts_start, ts_end])

targets = [
    "MetaINR",
    # "baseline",
    # "INR100",
    # "INR16",
]

# write the command to a csv
with open("tasks.csv", "w") as f:
    f.write("target,dataset,var,ts_range_st,ts_range_end\n")
    for target in targets:
        for (dataset, var), ts_ranges in rules.items():
            for ts_range in ts_ranges:
                f.write(f"{target},{dataset},{var},{ts_range[0]},{ts_range[1]}\n")
