#!/bin/bash
# JOB HEADERS HERE
#SBATCH -JTrainAttemptYOLORNN_ED512_B512                         # Job name
#SBATCH -N1 --gres=gpu:H200:1                       # Number of nodes and GPUs required
#SBATCH --mem-per-gpu=24G                           # Memory per gpu
#SBATCH -oReport-%j.out                             # Combined output and error messages file
#SBATCH -t480 # time (minutes; 480min = 8hrs walltime, unsure if that's amx)
#SBATCH --mail-type=BEGIN,END,FAIL                  # Mail preferences
#SBATCH --mail-user=ssingh709@gatech.edu            # e-mail address for notifications

module add anaconda3/2022.05 
# cd /home/hice1/<gt_username>/ondemand/<path to image_captioning subdirectory of repo>
cd "/home/hice1/ssingh709/scratch/CS4644-DL/src/image-captioning"
source activate SC4001 # venv
pip install wandb
srun python --version # checking that python is working
#srun python grid_train_script.py
# srun bash sample_train_command.sh # runinng train command file; modify that file per needs
srun bash wandbtrain.sh
