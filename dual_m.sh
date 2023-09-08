#!/bin/bash
# Made by Young Jin Kim (kimyoungjin06@gmail.com; kimyoungjin06@kentech.ac.kr)
# Last Update: 2022.12.01, YJ Kim
# This is the script for the Grid Search with slurm

# Example Script
# python3 GridSearch_slurm.py --multiproc=True --K_brdg_MAX=1 --P_brdg_MAX=1 --gamma_MAX=1 --K_brdg_Grid=11 --P_brdg_Grid=11 --gamma_Grid=11 --n_grid_init=4 --frequency_max=32 --out_path=20221129

##### This is slurm configuration #####
#SBATCH -J m=Mix   # job name
#SBATCH -n 1 # N of core(?)
#SBATCH -o dual_m.out   # standard output %j is job number

# echo "Jon on the %n-%2t$NODENUM node." ~/miniconda3/envs/latest/bin/python GridSearch_slurm.py
~/miniconda3/envs/latest/bin/python dual_m.py $1 $2 $3 $4 $5 $6 $7 $8