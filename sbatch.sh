#!/bin/bash
#SBATCH --job-name=TP_INNOV
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=00:03:00
#SBATCH --output=./logs/
#SBATCH --error=./logs/

srun ./Pytorch_PasswordGeneratorModelAI_V3.py