#!/bin/bash
# -M michael@yangm.tech
#$ -q gpu@qa-p100-* -l gpu=1
# -m e
#$ -r y
#$ -N fivejet

conda activate ml

python3 ../main.py --dataset=fivejet500 --replay

# qsub fivejet.sh