"""
usage: python3 GridSearch_slurm.py --K_brdg=<K_brdg> --P_brdg=<P_brdg> --gamma=<gamma> --n_grid_init=<n> --out_path=<output_path>

Update at 2022.11.29:

- With given K, P, and gamma, solve swing equation
- For the numerical calculation for initial condition dependency with given KPg params set
- Fix the initial theta(=0) because the only phase diffence is important.
- In this moment, we do not know how important the initial frequency or frequency difference.
  So, there are more sparse(sqrt(DoF(theta))) grid-search for conservation of degree of freedom in frequency.
- So, there is <degreen_grid_init>^2 Degree of Freedom.
"""

import numpy as np
import pandas as pd
import os
import swing
from datetime import datetime

today = default=datetime.today().strftime('%Y%m%d')

def get_res(K, P, gamma, n_grid_init, frequency_max, out_path):
    N = 2
    pi = np.pi
    pi2 = pi*2
    hpi = pi/2
    n = n_grid_init
    hn = int(np.sqrt(n_grid_init))
    fM = frequency_max
    
    t_end = 30.
    dt = .01
    t_eval = np.arange(0,t_end, dt)
    
    model = swing.wrapping.get_result_with_determider
    
    # space_t0g = np.linspace(0, pi2, n_grid_init, endpoint=False)
    space_t0c = np.linspace(0, pi2, n, endpoint=True)
    space_w0g = np.linspace(-fM, fM, hn, endpoint=True)
    space_w0c = np.linspace(-fM, fM, hn, endpoint=True)
    
    i = 0
    flat_inits = []
    t0g = 0.
    for t0c in space_t0c:
        for w0g in space_w0g:
            for w0c in space_w0c:
                swing_param = N, P, K, gamma, t_end, dt, t0g, w0g, t0c, w0c
                res = model(swing_param)
                flat_inits.append([t0c, w0g, w0c, res])

    return flat_inits

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Made by Young Jin Kim, 2022.11.25 (kimyoungjin06@gmail.com; kimyoungjin06@kentech.ac.kr)')
    parser.add_argument('--K_brdg', required=False, default=1, help='the value of strength of interaction between bridge nodes')
    parser.add_argument('--P_brdg', required=False, default=1, help='the value of Powers of bridge nodes (generator and consumer)')
    parser.add_argument('--gamma', required=False, default=1, help='the value of damping coefficient of system')
    parser.add_argument('--n_grid_init', required=False, default=256, help='the number of grid of initial values')
    parser.add_argument('--frequency_max', required=False, default=20, help='the number of grid of initial values')
    parser.add_argument('--out_path', required=False, default=today, help='the sub-dir of output feathers in Results/')
    parser.add_argument('--verbose', required=False, default=False, help='Show calculation times')
    args = parser.parse_args()
    
    K = float(args.K_brdg)
    P = float(args.P_brdg)
    gamma = float(args.gamma)
    n_grid_init = int(args.n_grid_init) + 1
    frequency_max = float(args.frequency_max)
    verbose = args.verbose
    out_path = f"Results/{args.out_path}"
    
    if verbose:
        from cProfile import Profile
        profiler = Profile()
        res = profiler.runcall(get_res, K, P, gamma, n_grid_init, frequency_max, out_path)
        # print(res)
        from pstats import Stats
        stats = Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        stats.print_stats(50)

    else:
        flat_inits = get_res(K, P, gamma, n_grid_init, frequency_max, out_path)
        Output_name = f"PD_state_{K:.2f}_{P:.2f}_{gamma:.2f}_{n_grid_init}"

        init_cols = ["theta_consumer_0", "omega_generator_0", "omega_consumer_0", "Value"]
        df = pd.DataFrame(flat_inits, columns=init_cols)
        cond_cols = ["K", "P/K", "gamma/sqrt(K)"]
        cond = [K, P/K, gamma/np.sqrt(K)]
        for i in range(len(cond)):
            df[cond_cols[i]] = cond[i]

        try:
            os.system(f"mkdir {out_path}")
        except:
            pass
        df[cond_cols + init_cols].to_feather(f"{out_path}/{Output_name}.ftr")
    