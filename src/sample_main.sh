#!/bin/bash
# JOB HEADERS HERE
#SBATCH -JTrainAttemptStack_ED512_BD128                         # Job name
#SBATCH -N1 --gres=gpu:RTX_6000:1                       # Number of nodes and GPUs required
#SBATCH --mem-per-gpu=12G                           # Memory per gpu
#SBATCH -oReport-%j.out                             # Combined output and error messages file
#SBATCH -t480 # time
#SBATCH --mail-type=BEGIN,END,FAIL                  # Mail preferences
#SBATCH --mail-user=<gt_username>@gatech.edu            # e-mail address for notifications

module add anaconda3/2022.05 
cd /home/hice1/<gt_username>/ondemand/<path to image_captioning subdirectory of repo>
source activate SC4001 # venv
srun python --version # checking that python is working
#srun python grid_train_script.py
srun bash sample_train_command.sh # runinng train command file; modify that file per needs
