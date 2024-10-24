#!/bin/bash
# -M michael@yangm.tech
#$ -q gpu@qa-p100-* -l gpu=1
# -m e
#$ -r y
#$ -N sub_one

conda activate ml

python3 ../run_task.py $1

# qsub sub_one.sh
