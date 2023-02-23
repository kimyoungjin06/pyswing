# swing/multiprocessing.py
"""
Note
-----
Made by Young Jin Kim (kimyoungjin06@gmail.com)
Last Update: 2022.11.09, YJ Kim

Use for multiprocessing in single node with multiple core.
- solve_func(params)
- multiprocess(P_max=2, K_max=10, P_grid=10, K_grid=10, 
                     t_end=30., dt=.001, n_cpu=19)
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool
import time

from . import core
from . import initialize


# For solving
### To Do
# - Unifying Function
# def solver(func, params, init_func=None, utilize_func=None, intergrate_func):
#     """
#     Note
#     ----
#     Solving with multiprocess().
    
#     Parameter
#     ---------
#     params : dict
#         Contains all parameter-set.
#     """
#     # Extract from params
#     ## For Initializing
#     N = params['N']
#     Lambda = params['Lambda'] # For SF, 1 < Lambda < 3
#     degree_type = params['degree_type'] # or SF, ER
#     MAXDegree = params['MAXDegree'] # For Lattice, MAXDegree != 0
#     Backward = params['Backward']
#     zero_mean_power = params['zero_mean_power']
#     esl = params['esl'] # For SF, Add to Lambda for when gamma=1
    
#     Initialize = swing.annealed_multilayer.Init_Anneal
#     inits = Initialize(N=N, Backward=Backward, zero_mean_power=zero_mean_power, 
#                        degree_type=degree_type, MAXDegree=MAXDgree, Lambda=Lambda, esl=esl)
    
#     ## For a configuration
#     K = params['K']
#     M1 = params['M1']
#     COEF_GAMMA = params['COEF_GAMMA']
#     t_end = params['t_end']
#     dt = params['dt']
#     target_time = params['target_time']
def try_make_folder(parent, offspring):
    import os
    if offspring.split('/')[-1] not in os.listdir(parent):
        try:
            os.system(f"mkdir {offspring}")
        except:
            pass

def PrintProfile(num_stats, func, params):
    from cProfile import Profile
    profiler = Profile()
    res = profiler.runcall(func, params)
    # print(res)
    from pstats import Stats
    stats = Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(num_stats)


def Run_Single(kwarg):
    """
    Run_Single([kwarg])
    """
    model, params, verbose, num_stats, out_path = kwarg
    # Log parameters
    pd.DataFrame(params, index=[0]).to_feather(f"{out_path}/parameters.ftr")

    N = params['N']
    K = params['K']
    Ensemble = params['Ensemble']
    COEF_GAMMA = params['COEF_GAMMA']
    if params['Backward']:
        B = 'Backward'
    else:
        B = 'Forward'
    # CONST_dt = params['CONST_dt']
    Output_name = f"temp_res_N{N}_K{K:.2f}_G{COEF_GAMMA}_{B}_E{Ensemble}"  #_CONST_dt{CONST_dt}"

    # # There are already result The code is terminated
    # if f"{Output_name}.ftr" in os.listdir(f"{out_path}"):
    #     return 0

    if verbose:
        # Print upto 50th profiles
        PrintProfile(num_stats, model, params)
    else:
        res = []
        for Ei in range(Ensemble):
            _res = model(params)
            res.append(_res)

        # If there are no folder
        pd.DataFrame(res, columns=['ensemble']).to_feather(f"{out_path}/{Output_name}.ftr")


def Run_Multiple(model, params):
    """
    Run_Multiple(params)
    """
    from multiprocessing import Pool
    import time

    start = int(time.time())
    pd.DataFrame(params, index=[0]).to_feather(f"{out_path}/parameters.ftr")

    K_stt = params['K_stt']
    K_end = params['K_end'] + 0.0001
    dK = params['dK']
    N_CPU = params['N_CPU']
    K_space = np.arange(K_stt, K_end, dK)

    num_stats = 50
    verbose = False
    # Get Params
    paramss = []
    for Backward in [True, False]:
        for K in K_space:
            params['Backward'] = Backward
            params['K'] = K
            _params = model, params.copy(), verbose, num_stats
            paramss.append(_params)

    p = Pool(processes=N_CPU)
    result = p.map(Run_Single, paramss)

#     df = pd.DataFrame(result)
#     df['K'] = np.append(K_space,K_space)
#     df['Backward'] = [True]*len(K_space) + [False]*len(K_space)
#     df.to_feather(f"{out_path}/res_tot.ftr")

    end = int(time.time())
    print("Number of Core : " + str(N_CPU))
    print("***run time(min) : ", (end-start)/60.)


def solve_func(params):
    """
    Note
    ----
    Solving with multiprocess().
    
    Parameter
    ---------
    params : dict
        Contains all parameter-set.
    """
    
    # Extract from params
    SP = params['sparam']
    t_end = params['t_end']
    y0 = params['init']
    dt = params['dt']
    
    # Extract from Swing_params (SP)
    network = SP["network"]
    m = SP["m"]
    gamma = SP["gamma"]
    P = SP["P"]
    K = SP["K"]
    # _model = SP["model"]
    
    t_eval = np.arange(0,t_end, dt)

    # integrate with lsoda method
    # rhs = swing.swing_lsoda(network, N)
    # funcptr = rhs.address
    # solution, success = lsoda(funcptr, y0, t_eval)
    
    # Integrate with RK4
    solution = core.RK4(swing, t_end, y0, dt,
                   m, gamma, P, K, 
                   network)
    return solution

def solve_func_with_det(params):
    """
    Note
    ----
    Solving the swing equation with single parameter.
    
    
    Parameter
    ---------
    params : dict
        Contains all parameter-set.
        
    Return
    ------
    Determiner for Phase Diagram
    """
    
    # Extract from params
    SP = params['sparam']
    N = params['N']
    t_end = params['t_end']
    y0 = params['init']
    dt = params['dt']
    
    # Extract from Swing_params (SP)
    network = SP["network"]
    m = SP["m"]
    gamma = SP["gamma"]
    P = SP["P"] # This P is list of all node
    K = SP["K"]
    # _model = SP["model"]
    
    P = P[N-1]
    NN = N*2 # Total N for both sides
    t_eval = np.arange(0,t_end, dt)
    
    # Integrate with RK4
    solution = core.RK4(swing, t_end, y0, dt,
                   m, gamma, P, K, 
                   network)
    
    # Get Determiner
    res = np.array(res)
    res2 = res.reshape(t_eval.shape[0], 2, NN) # 2 means theta, omega
    theta = res2[:,0,:] # t, 0, node_idx
    omega = res2[:,1,:] # t, 1, node_idx
    det = determiner(omega, N)
    
    return det

# For Multiprocessing on a single-CPU

def multiprocess_GridSearch(P_max=10., P_grid=11, 
                            K_max=10., K_grid=11, 
                            gamma_max=1., gamma_grid=11,
                            N=3, t0g=0, w0g=0, t0c=0, w0c=0,
                            t_end=30., dt=.005, n_cpu=19):
    """
    Note
    ----
    For Grid Searching to obtain full results.
    
    Parameter
    ---------
    P_max : float64
        Max values of Power
    P_grid : integer
        Number of grid of Power
    K_max : float64
        
    K_grid : integer
    t_end : float64
        End of time for intergration.
    dt : float64
        Unit of time for intergration.
    n_cpu : integer
        Number of CPU for using in single-node cluster.
    """
    start = int(time.time())
    paramss = []
    
    space_P = np.linspace(0, P_max, P_grid)
    space_K = np.linspace(0, K_max, K_grid)
    space_g = np.linspace(0, gamma_max, gamma_grid)
    for P in space_P:
        for K in space_K:
            for K in space_g:
                # initialization(N, P, K, gamma=2.0, t0g=0, w0g=0, t0c=0, w0c=0)
                Swing_Parameters, y0 = initialize(N, P, K, gamma, 
                                                  t0g, w0g, t0c, w0c)

                params = {}
                params['sparam'] = Swing_Parameters
                params['t_end'] = t_end
                params['init'] = y0
                params['dt'] = dt
                paramss.append(params)
                del([[params]])

    p = Pool(processes=n_cpu)
    result = p.map(solve_func, paramss)
    
    end = int(time.time())
    print("***run time(sec) : ", end-start)
    print("Number of Core : " + str(n_cpu))
    return result

def multiprocess_GridSearch_withDet(P_max=3., P_grid=11, 
                            K_max=10., K_grid=11, 
                            gamma_max=1., gamma_grid=11,
                            N=3, t0g=0, w0g=0, t0c=0, w0c=0,
                            t_end=30., dt=.005, norm=False, n_cpu=19):
    """
    Note
    ----
    For Grid Searching to get Phase Diagram with Determiner.
    
    Parameter
    ---------
    P_max : float64
        Max values of Power
    P_grid : integer
        Number of grid of Power
    K_max : float64
        
    K_grid : integer
    t_end : float64
        End of time for intergration.
    dt : float64
        Unit of time for intergration.
    n_cpu : integer
        Number of CPU for using in single-node cluster.
    """
    start = int(time.time())
    paramss = []
    
    
    space_K = np.linspace(1., K_max, K_grid)
    space_P = np.linspace(0., P_max, P_grid)
    space_g = np.linspace(0., gamma_max, gamma_grid)
    for K in space_K:
        # Convert P -> P/K, gamma -> gamma/sqrt(K)
        if norm:
            space_P = np.linspace(0., P_max*K, P_grid)
            space_g = np.linspace(0., gamma_max*np.sqrt(K), gamma_grid)
        for P in space_P:
            for gamma in space_g:
                # initialization(N, P, K, gamma=2.0, t0g=0, w0g=0, t0c=0, w0c=0)
                Swing_Parameters, y0, _ = initialization(N, P, K, gamma, t0g, w0g, t0c, w0c)

                params = {}
                params['sparam'] = Swing_Parameters
                params['N'] = N
                params['t_end'] = t_end
                params['init'] = y0
                params['dt'] = dt
                paramss.append(params)
                del([[params]])
        del([[space_P, space_g]])

    p = Pool(processes=n_cpu)
    result = p.map(solve_func_with_det, paramss)
    
    end = int(time.time())
    print("Number of Core : " + str(n_cpu))
    print("***run time(min) : ", (end-start)/60.)
    return result