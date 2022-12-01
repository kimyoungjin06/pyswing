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
Description = """
--------------------
Made by Young Jin Kim, 2022.11.29 (kimyoungjin06@gmail.com; kimyoungjin06@kentech.ac.kr)

For Example,
i) Multiprocess
> python3 GridSearch_slurm.py --multiproc=True --K_brdg_MAX=1 --P_brdg_MAX=1 --gamma_MAX=1 --K_brdg_Grid=11 --P_brdg_Grid=11 --gamma_Grid=11 --n_grid_init=32 --frequency_max=32 --out_path=20221129

ii) Single-running
> python3 GridSearch_slurm.py --K_brdg=1 --P_brdg=1 --gamma=1 --n_grid_init=32 --frequency_max=32 --verbose=True
--------------------
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

def PrintProfile(num_stats, K, P, gamma, n_grid_init, frequency_max, out_path):
    from cProfile import Profile
    profiler = Profile()
    res = profiler.runcall(get_res, K, P, gamma, n_grid_init, frequency_max, out_path)
    # print(res)
    from pstats import Stats
    stats = Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(num_stats)

def Run_Single(params):
    """
    Run_Single([50, K, P, gamma, n_grid_init, frequency_max, out_path, verbose])
    """
    num_stats, K, P, gamma, n_grid_init, frequency_max, out_path, verbose = params
    
    Output_name = f"PD_state_{K:.2f}_{P:.2f}_{gamma:.2f}_{n_grid_init}"
    
    # There are already result The code is terminated
    if f"{Output_name}.ftr" in os.listdir(f"{out_path}"):
        return 0
    
    if verbose:
        # Print upto 50th profiles
        PrintProfile(num_stats, K, P, gamma, n_grid_init, frequency_max, out_path)

    else:
        flat_inits = get_res(K, P, gamma, n_grid_init, frequency_max, out_path)
        

        init_cols = ["theta_consumer_0", "omega_generator_0", "omega_consumer_0", "Value"]
        df = pd.DataFrame(flat_inits, columns=init_cols)
        cond_cols = ["K", "P/K", "gamma/sqrt(K)"]
        cond = [K, P/K, gamma/np.sqrt(K)]
        for i in range(len(cond)):
            df[cond_cols[i]] = cond[i]

        # If there are no folder
        if out_path.split('/')[-1] not in os.listdir("Results/"):
            os.system(f"mkdir {out_path}")

        df[cond_cols + init_cols].to_feather(f"{out_path}/{Output_name}.ftr")
        
def Run_Multiple(K_max, K_grid, P_max, P_grid, gamma_max, gamma_grid, n_grid_init, frequency_max, out_path, Norm, N_CPU):
    """
    Run_Multiple(K_max, K_grid, P_max, P_grid, gamma_max, gamma_grid, n_grid_init, frequency_max, out_path, N_CPU)
    """
    
    from multiprocessing import Pool
    import time
    
    start = int(time.time())
    
    # Get Params
    paramss = []
    space_K = np.linspace(1., K_max, K_grid)
    # space_P = np.linspace(0., P_max, P_grid)
    # space_g = np.linspace(0., gamma_max, gamma_grid)
    for K in space_K:
        # Convert P -> P/K, gamma -> gamma/sqrt(K)
        if Norm:
            space_P = np.linspace(0., P_max*K, P_grid)
            space_g = np.linspace(0., gamma_max*np.sqrt(K), gamma_grid)
        
        for P in space_P:
            for gamma in space_g:
                num_stats = 0 # For dummy
                verbose = False
                params = num_stats, K, P, gamma, n_grid_init, frequency_max, out_path, verbose
                paramss.append(params)
                del([[params]])
        del([[space_P, space_g]])

    p = Pool(processes=N_CPU)
    result = p.map(Run_Single, paramss)
    
    end = int(time.time())
    print("Number of Core : " + str(N_CPU))
    print("***run time(min) : ", (end-start)/60.)
    
def Run_Multiple(K_max, K_grid, P_max, P_grid, gamma_max, gamma_grid, n_grid_init, frequency_max, out_path, Norm, N_CPU):
    """
    Run_Multiple(K_max, K_grid, P_max, P_grid, gamma_max, gamma_grid, n_grid_init, frequency_max, out_path, N_CPU)
    """
    
    from multiprocessing import Pool
    import time
    
    start = int(time.time())
    
    # Get Params
    paramss = []
    space_K = np.linspace(1., K_max, K_grid)
    # space_P = np.linspace(0., P_max, P_grid)
    # space_g = np.linspace(0., gamma_max, gamma_grid)
    for K in space_K:
        # Convert P -> P/K, gamma -> gamma/sqrt(K)
        if Norm:
            space_P = np.linspace(0., P_max*K, P_grid)
            space_g = np.linspace(0., gamma_max*np.sqrt(K), gamma_grid)
        
        for P in space_P:
            for gamma in space_g:
                num_stats = 0 # For dummy
                verbose = False
                params = num_stats, K, P, gamma, n_grid_init, frequency_max, out_path, verbose
                paramss.append(params)
                del([[params]])
        del([[space_P, space_g]])

    p = Pool(processes=N_CPU)
    result = p.map(Run_Single, paramss)
    
    end = int(time.time())
    print("Number of Core : " + str(N_CPU))
    print("***run time(min) : ", (end-start)/60.)
    
if __name__ == '__main__':
    # Extract argparse
    import argparse
    parser = argparse.ArgumentParser(description=Description, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--multiproc', required=False, default=False, help='GridSerch or Single-runinng')
    
    ## Related with Multiprocess
    parser.add_argument('--K_brdg_MAX', required=False, default=20, help='For GridSerch, the Maximum value of strength of interaction between bridge nodes')
    parser.add_argument('--K_brdg_Grid', required=False, default=11, help='For GridSerch, the number of grid of strength of interaction between bridge nodes')
    parser.add_argument('--P_brdg_MAX', required=False, default=10, help='For GridSerch, the Maximum value of Powers of bridge nodes (generator and consumer)')
    parser.add_argument('--P_brdg_Grid', required=False, default=11, help='For GridSerch, the number of grid of Powers of bridge nodes (generator and consumer)')
    parser.add_argument('--gamma_MAX', required=False, default=10, help='For GridSerch, the Maximum value of damping coefficient of system')
    parser.add_argument('--gamma_Grid', required=False, default=11, help='For GridSerch, the number of grid of damping coefficient of system')
    parser.add_argument('--n_CPU', required=False, default=8, help='For GridSerch, the total number of parallel jobs')
    
    ## Related with Single-running
    parser.add_argument('--K_brdg', required=False, default=1, help='For Single-running, the value of strength of interaction between bridge nodes')
    parser.add_argument('--P_brdg', required=False, default=1, help='For Single-running, the value of Powers of bridge nodes (generator and consumer)')
    parser.add_argument('--gamma', required=False, default=1, help='For Single-running, the value of damping coefficient of system')
    
    ## Related with initial conditions
    parser.add_argument('--n_grid_init', required=False, default=256, help='the number of grid of initial values')
    parser.add_argument('--frequency_max', required=False, default=20, help='the number of grid of initial values')
    
    ## ETC
    parser.add_argument('--out_path', required=False, default=today, help='the sub-dir of output feathers in Results/')
    parser.add_argument('--verbose', required=False, default=False, help='For Single-running, Show calculation times')
    
    args = parser.parse_args()
    ## Set common args
    MultiProc = args.multiproc
    out_path = f"Results/{args.out_path}"
    n_grid_init = int(args.n_grid_init) + 1
    frequency_max = float(args.frequency_max)
    
    # If there are no folder
    if out_path.split('/')[-1] not in os.listdir("Results/"):
        os.system(f"mkdir {out_path}")
        
    if MultiProc:
        # Init args
        K_max = float(args.K_brdg_MAX)
        P_max = float(args.P_brdg_MAX)
        gamma_max = float(args.gamma_MAX)
        
        K_grid = int(args.K_brdg_Grid)
        P_grid = int(args.P_brdg_Grid)
        gamma_grid = int(args.gamma_Grid)
        Norm = True # P -> P/K, gamma ->gamma/sqrt(K)
        
        N_CPU = int(args.n_CPU)
        
        # Run Multiple Process
        Run_Multiple(K_max, K_grid, P_max, P_grid, gamma_max, gamma_grid, 
                     n_grid_init, frequency_max, out_path, Norm, N_CPU)
        
    else:
        # Init args
        K = float(args.K_brdg)
        P = float(args.P_brdg)
        gamma = float(args.gamma)
        
        verbose = args.verbose
        num_stats = 50

        # Run single
        params = num_stats, K, P, gamma, n_grid_init, frequency_max, out_path, verbose
        Run_Single(params)
