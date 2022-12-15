# swing/minor.py
"""
Note
-----
Made by Young Jin Kim (kimyoungjin06@gmail.com)
Last Update: 2022.11.24, YJ Kim

Some functions related with Network Translation and Phase Diagram
"""

import numpy as np

# Related with Network Translation

def AdjMtoAdjL(adjM: np.ndarray) -> list:
    """ Adjacency Matrix to Adjacency List"""
    return [np.argwhere(adjM[:,i] > 0).flatten() for i in range(len(adjM))]
def AdjMtoEdgL(adjM: np.ndarray) -> np.ndarray:
    """ Adjacency Matrix to Edge List"""
    return np.argwhere(adjM > 0)

# Related with Drawing Phase Diagram

def determiner(omega, N):
    """
    Note
    ----
    Determine the Phase of result.
    For drawing phase diagram of Double-clique(or layer).
    The main point is what node is synchronized with the bridge nodes.
    
    Determiner returns a values which is 
    the ration of OTM (Omega Time Mean) of bridge node and leaf node.
    $Det = \langle O_{brdg} \rangle _{t} / \langle O_{leaf} \rangle _{t}$
    
    Parameter
    ---------
    omega : Contains all parameter-set.
    N : Number of Nodes in a side
    
    Return
    ------
    ratio of OTM of bridge and leaf.
    """
    #Omega Time Mean
    OTM = omega.mean(axis=0)
    OTM_brdg = np.abs(OTM[N-1:N+1]).mean()
    OTM_leaf = np.abs([OTM[:N-1].mean(), OTM[N+1:].mean()]).mean()
    det = OTM_brdg / OTM_leaf
    return det

# Log-log Fitting

def plfunc(x, g, C, x0, y0):
    """
    note
    ----
    Fitting function with $y = C*(x-x0)^(-G) + y0$.
    """
    return C*(x-x0)**(-g) + y0

def log_log_fit(func,x,y,**kwargs):
    """
    note
    ----
    Fitting with log.
    """
    import scipy
    
    # Masking Positive bins
    msk = y > 0
    x = x[msk]
    y = y[msk]
    
    def f_conv(x, a, b, c, d):
        return np.log(func(np.exp(x), a, b, c, d))
    
    log_x, log_y = np.log(x),np.log(y)
    
    return scipy.optimize.curve_fit(f_conv, log_x, log_y,**kwargs)