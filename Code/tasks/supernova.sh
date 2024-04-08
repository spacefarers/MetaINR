#!/bin/bash
# -M michael@yangm.tech
#$ -q gpu@qa-p100-* -l gpu=1
# -m e
#$ -r y
#$ -N supernova

conda activate ml

python3 ../main.py --dataset=supernova --replay

# qsub supernova.sh