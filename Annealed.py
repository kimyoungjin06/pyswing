"""
usage: python3 Annealed.py --K_stt=<K_stt> --K_end=<K_end> --dK=<dK> --Ensemble=<Ensemble> --N_CPU=<N_CPU>

Update at 2023.2.21:
- Add Regular Sampling with ["Gaussian", "Cauchy", "PowerLaw", "Uniform"]

Update at 2023.1.26:
- Single Layer with Annealed Approximation swing equation solver with RK4
"""
Description = """
--------------------
Made by Young Jin Kim, 2023.1.26 (kimyoungjin06@gmail.com; kimyoungjin06@kentech.ac.kr)

For Example,
i) Multiprocess
> python3 Annealed.py --K_stt=<K_stt> --K_end=<K_end> --dK=<dK> --Ensemble=<Ensemble> --N_CPU=<N_CPU> --multiproc=True

ii) Single-running
> python3 Annealed.py --K_stt=<K_stt> --K_end=<K_end> --dK=<dK> --Ensemble=<Ensemble> --verbose=True
--------------------
"""

import numpy as np
import pandas as pd
import os
import pyswing.swing as swing
from datetime import datetime

today = datetime.today().strftime('%Y%m%d')

params = {}

### Default Values
# For Inits
params['N'] = 3000  # 1024*(2**4)
params['degree_type'] = 'FC'  # or SF, ER
params["degree_exp"] = 3.0  # For SF, 1 < Lambda < 3
params['MINDegree'] = 5  # For Lattice, MAXDegree != 0
params['MAXDegree'] = 0  # For Lattice, MAXDegree != 0
params["power_type"] = "Gaussian"  # ['Gaussian', 'Cauchy', 'PowerLaw', 'Uniform']
params["power_exp"] = 1.0
params["MINPower"] = 5  # For Lattice, MAXDegree != 0
params["MAXPower"] = 0  # For Lattice, MAXDegree != 0
params["RegularSampling"] = True
params['Backward'] = False
params['zero_mean_power'] = True
params['esl'] = 1E-2  # For SF, Add to Lambda for when gamma=1

# For a configuration
params['K'] = 1.
params['M1'] = 1.
params['COEF_GAMMA'] = .4
params["t_end"] = 400.0
params["dt"] = 0.002
params['target_time'] = 2.
params['ReduceMemory'] = True

# For Loop Information
params['Loops'] = "K,COEF_GAMMA,Backward"
params['K.max'] = 20.0
params['K.min'] = 0.1
params['K.d'] = 0.1
params['K.type'] = 'float'
params['COEF_GAMMA.series'] = '0.001 0.01 0.1 0.2 0.4'# 0.8 1.6'
params['COEF_GAMMA.type'] = 'float'
params['Ensemble'] = 10
params['out_path'] = 'FC120_3000'  # 'FC120_3000'
params['Description'] = 'Long t_end, small dt, gamma-defendent, and Add Ensemble'
# params['CONST_dt'] = 1.
# params[''] = 

if __name__ == '__main__':
    # Extract argparse
    import argparse
    parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--multiproc', required=False, default=False, help='GridSerch(True) or Single-runinng(False)')

    ## Related with Multiprocess
    parser.add_argument('--K_stt', required=False, default=.1, help='For GridSerch, the Minimum value of strength of interaction')
    parser.add_argument('--K_end', required=False, default=4., help='For GridSerch, the Maximum value of strength of interaction')
    parser.add_argument('--dK', required=False, default=10, help='For GridSerch, the number of grid of strength of interaction')
    parser.add_argument('--Ensemble', required=False, default=10, help='For GridSerch, the number of ensemble')
    parser.add_argument('--N_CPU', required=False, default=8, help='For GridSerch, the total number of parallel jobs')

    ## Related with SingleRunning
    parser.add_argument('--K', required=False, default=.1, help='For Single running, the value of strength of interaction')
    parser.add_argument('--Backward', required=False, default=False, help='For Single running, the initial condition of phases')
    parser.add_argument('--COEF_GAMMA', required=False, default=False, help='For Single running, the COEF_GAMMA')
    # parser.add_argument('--CONST_dt', required=False, default=False, help='For Single running, the time ratio constant to determine node locking state in getting orderparameter')

    ## ETC
    parser.add_argument('--out_path', required=False, default=today, help='the sub-dir of output feathers in Results/')
    parser.add_argument('--verbose', required=False, default=False, help='For Single-running, Show calculation times')

    args = parser.parse_args()

    ## Set common args
    MultiProc = args.multiproc
    parent = 'Results'
    if 'out_path' in params:
        out_path = f'{parent}/{params["out_path"]}'
    else:
        out_path = f'{parent}/{args.out_path}'
    model = swing.annealed_multilayer.solve_func_with_orderparam2

    # If there are no folder
    swing.multiprocessing.try_make_folder(parent, out_path)

    if MultiProc:
        from pyswing.swing.multiprocessing import Run_Multiple
        # Init args
        params['K_stt'] = float(args.K_stt)
        params['K_end'] = float(args.K_end) + 0.0001
        params['dK'] = float(args.dK)
        params['Ensemble'] = int(args.Ensemble)
        params['N_CPU'] = int(args.N_CPU)

        # Run Multiple Process
        Run_Multiple(model, params)
    else:
        from pyswing.swing.multiprocessing import Run_Single
        # Init args
        verbose = args.verbose
        num_stats = 50
        params['K'] = float(args.K)

        if args.Backward == 'True':
            params['Backward'] = True
        else:
            params['Backward'] = False
        if args.Ensemble:
            params['Ensemble'] = int(args.Ensemble)
        # if args.CONST_dt:
        #     params['CONST_dt'] = float(args.CONST_dt)

        # Run single
        kwarg = [model, params, verbose, num_stats, out_path]
        Run_Single(kwarg)