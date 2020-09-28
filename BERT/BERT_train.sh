#!/bin/bash

#SBATCH -p condo
#SBATCH --gres=gpu:1
#SBATCH --reservation=condo_serdal_85
#SBATCH -n 16

#SBATCH -t 48:00:00
#SBATCH --mem=100GB

module purge
source ~/.bashrc
source activate openml

python ./BERT_train.py
