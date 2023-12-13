#!/bin/bash
#SBATCH --job-name=TP_INNOV
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --gpus-per-node=3
#SBATCH --partition=gpu
#SBATCH --time=00:59:00

python3 ./Pytorch_PasswordGeneratorModelAI_V3.py

