#!/bin/bash
# -M michael@yangm.tech
#$ -q gpu@qa-p100-* -l gpu=1
# -m e
#$ -r y
#$ -N inr

conda activate ml

#python3 ../INR_encoding.py --dataset=fivejet500 --train_iterations=500
python3 ../INR_encoding.py --dataset=supernova --train_iterations=500
#python3 ../INR_encoding.py --dataset=tangaroa --train_iterations=500

# qsub inr.sh