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
from multiprocessing import Pool
import time

from .core import *
from .initialize import *



# For solving

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
    solution = RK4(swing, t_end, y0, dt,
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
    solution = RK4(swing, t_end, y0, dt,
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