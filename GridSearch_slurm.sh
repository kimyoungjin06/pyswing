#!/bin/bash

# Made by Young Jin Kim (kimyoungjin06@gmail.com; kimyoungjin06@kentech.ac.kr)
# Last Update: 2022.11.30, YJ Kim
# This is the script for the Grid Search with slurm

# Example Script
# python3 GridSearch_slurm.py --multiproc=True --K_brdg_MAX=1 --P_brdg_MAX=1 --gamma_MAX=1 --K_brdg_Grid=11 --P_brdg_Grid=11 --gamma_Grid=11 --n_grid_init=4 --frequency_max=32 --out_path=20221129

##### This is slurm configuration #####
#SBATCH -J PhaDgr_N4   # job name
#SBATCH -n 5000000 # N of core
#SBATCH -o 20221201.out   # standard output %j is job number

n_grid=32
fmax=32

for K in $(seq 0 1 5); do
    for P in $(seq 0 0.1 5); do
        for g in $(seq 0 0.1 5); do
            python3 GridSearch_slurm.py --K_brdg=$K --P_brdg=$P --gamma=$g --n_grid_init=$n_grid --frequency_max=$fmax
        done
    done
done