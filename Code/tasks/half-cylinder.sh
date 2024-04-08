#!/bin/bash
# -M michael@yangm.tech
#$ -q gpu@qa-p100-* -l gpu=1
# -m e
#$ -r y
#$ -N half-cylinder

conda activate ml

python3 ../main.py --dataset=half-cylinder --var="640" --pretrain

# qsub half-cylinder.sh