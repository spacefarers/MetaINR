#!/bin/bash
# -M michael@yangm.tech
#$ -q gpu@qa-p100-* -l gpu=1
# -m e
#$ -r y
#$ -N tangaroa

conda activate ml

python3 ../main.py --dataset=tangaroa --replay

# qsub tangaroa.sh