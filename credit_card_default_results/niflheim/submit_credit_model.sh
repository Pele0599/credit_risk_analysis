#!/usr/bin/bash 

#SBARCH --job-name=mdsim
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s183798@student.dtu.dk  
#SBATCH --partition=xeon16
#SBATCH -N 1      
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
source ~/anaconda3/etc/profile.d/conda.sh
conda activate srbench

python3 /home/energy/pvifr/cred_anal/credit_risk_analysis/credit_card_default_results/train_gplearn_niflheim.py

