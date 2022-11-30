#!/bin/bash

# Made by Young Jin Kim (kimyoungjin06@gmail.com; kimyoungjin06@kentech.ac.kr)
# Last Update: 2022.11.30, YJ Kim
# This is the script for the Grid Search with slurm

# Example Script
# python3 GridSearch_slurm.py --multiproc=True --K_brdg_MAX=1 --P_brdg_MAX=1 --gamma_MAX=1 --K_brdg_Grid=11 --P_brdg_Grid=11 --gamma_Grid=11 --n_grid_init=4 --frequency_max=32 --out_path=20221129


#SBATCH -J PhaseDiagram_N4   # job name
#SBATCH -o 20221130.%j.out   # standard output and error log
#SBATCH -p PDfN4           # queue  name  or  partiton name
#SBTACH --ntasks=121 # The Number of jobs

python3 GridSearch_slurm.py --multiproc=True --K_brdg_MAX=1 --P_brdg_MAX=1 --gamma_MAX=1 --K_brdg_Grid=11 --P_brdg_Grid=11 --gamma_Grid=11 --n_grid_init=4 --frequency_max=32 --out_path=20221129