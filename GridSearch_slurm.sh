#!/bin/bash
# Made by Young Jin Kim (kimyoungjin06@gmail.com; kimyoungjin06@kentech.ac.kr)
# Last Update: 2022.12.01, YJ Kim
# This is the script for the Grid Search with slurm

# Example Script
# python3 GridSearch_slurm.py --multiproc=True --K_brdg_MAX=1 --P_brdg_MAX=1 --gamma_MAX=1 --K_brdg_Grid=11 --P_brdg_Grid=11 --gamma_Grid=11 --n_grid_init=4 --frequency_max=32 --out_path=20221129

Kmin=0.025
Kmax=15.0
dK=0.025
Ensemble=1
Forward=False
Backward=True
dts="$(echo 0.001, 0.01 0.1 0.2 0.4 0.8 1.6)"

for COEF_GAMMA in $dts; do
    for K in $(seq 0.1 $dK $Kmax); do
        sbatch GridSearch_slurm_node2.sh --K=$K --Ensemble=$Ensemble --COEF_GAMMA=$COEF_GAMMA --Backward=$Backward 
    done

    for K in $(seq 0.1 $dK $Kmax); do
        sbatch GridSearch_slurm_node2.sh --K=$K --Ensemble=$Ensemble --COEF_GAMMA=$COEF_GAMMA --Backward=$Forward --out_path=$Forward
    done
done
