"""
This script is used to run a task specified by the task number.
The task number is used to index the tasks/tasks.csv file.
Useful for submitting tasks to a cluster or running multiple tasks in parallel.
"""

import fire
import config
import main
import INR_encoding
import baseline

def run_a_task(task_number):
    config.enable_logging = True
    # open tasks/tasks.csv and read the task_number-th line
    task_number -= 1
    with open("tasks.csv", "r") as f:
        lines = f.readlines()
        target, dataset, var, ts_range_st, ts_range_end = lines[task_number].strip().split(",")
    ts_range = (int(ts_range_st), int(ts_range_end))
    config.log({"target": target, "dataset": dataset, "var": var, "ts_range": str(ts_range)})
    if target == "baseline":
        baseline.run(dataset=dataset, var=var, ts_range=ts_range)
    elif target == "INR":
        INR_encoding.run(dataset=dataset, var=var, ts_range=ts_range)
    elif target == "MetaINR":
        main.run(dataset=dataset, var=var, ts_range=ts_range)


if __name__ == '__main__':
    fire.Fire(run_a_task)
