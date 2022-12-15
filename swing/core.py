# swing/core.py
"""
Note
-----
Made by Young Jin Kim (kimyoungjin06@gmail.com)
Last Update: 2022.11.09, YJ Kim

Functions
---------
Core Functions for ODE
- SinIntE;
    Interaction Term with numba nopython.
- swing;
    Swing Equation on the network.
- RK4;
    Runge-Kutta 4th order.
    - Need to speed optimization with numba.
"""

__all__ = ['SinIntE', 'RK4', 'swing',]

import numpy as np
import numba as nb

# Core Functions for Swing

@nb.jit(nopython=True)
def SinIntE(EdgeList: np.array([[]]), T: np.ndarray, K: np.array([[]])):
    """
    Note
    ----
    Calculate with Sin Interaction term with Edge List.
    - Need to speed optimize for more efficiency.
    """
    N = EdgeList[:,0].max() + 1 # Start from 0
    sinint = np.zeros(N, dtype=np.float64)
    for i, j in EdgeList:
        if i < j:
            _int = K[i,j]*np.sin(T[j] - T[i])
            sinint[i] += _int
            sinint[j] -= _int
    return sinint

def swing(t, y, m, gamma, P, K, network) -> np.array([[]]):
    """
    \dot{\theta} &= \omega \\
    \dot{\omega} &= \frac{1}{m}(P-\gamma\omega+\Sigma K\sin(\theta-\phi))
    """
    T, O = y
    
    m = np.array(m)
    P = np.array(P)

    # Get Interaction
#     Interaction = K*SinIntCover(net_addr, net_shape, net_dtype, T)
    Interaction = SinIntE(network, T, K)
    dT = O
    dO = 1/m*(P - gamma*O + Interaction)
    dydt = np.concatenate(([dT], [dO]))#, dtype=np.float64)
    return dydt

def RK4(func:np.array, t_end, X0, dt, m, gamma, P, K, network, *kwargs):
    """
    Note
    ----
    Runge-Kutta with 4th order.
    Normally, enough $dt=0.05$ for swing equation on the network.
    
    - Backlog
        - Need to speed optimization with numba.
        - Update to possible localized gamma.

    ...

    Attributes
    ----------
    func : 1d-ndarray
        A formatted string to print out what the animal says
    t_end : float64
        The end of time for iteration
    X0 : (N,2) 2d-ndarray
        Initial values of ODE problem
    dt : float64
        Unit of time
    m : float64
        Intertia of nodes
    gamma : float64
        Damping of system (global) or each node (local constants)
    P : List(float46)
        Intrinsic powers of each node with size N.
    K : 2d-ndarray
        Adjacency Matrix with weighted network.
    network : (N(E),2) 2d-ndarray
        Edge list of network with size N(E)

    Methods
    -------
    RK4(swing.core.swing, t_end, X0, dt=0.05, 
        Inertias, Gamma, Powers, K, 
        Network)
    """
    t = np.arange(0., t_end, dt)
    X  = np.zeros((t.shape[0], X0.shape[0], X0.shape[1]))
    X[0] = X0
    hdt = dt*.5
    for i in range(t.shape[0]-1):
        t1 = t[i]
        x1 = X[i]
        k1 = func(t[i], X[i], m, gamma, P, K, network, *kwargs)
        
        t2 = t[i] + hdt
        x2 = X[i] + hdt * k1
        k2 = func(t2, x2, m, gamma, P, K, network, *kwargs)
        
        t3 = t[i] + hdt
        x3 = X[i] + hdt * k2
        k3 = func(t3, x3, m, gamma, P, K, network, *kwargs)
        
        t4 = t[i] + dt
        x4 = X[i] + dt * k3
        k4 = func(t4, x4, m, gamma, P, K, network, *kwargs)
        X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    return X

########################## In Backlog ########################
# from numbalsoda import lsoda_sig, lsoda, dop853

###### RK4 with Numba nopython ######
# @nb.jit(nopython=True)
# def RK4(func, t_end, X0, dt, m, gamma, P, K, net_addr, net_shape, net_dtype):
    
#     t = np.arange(0,t_end, dt, dtype=np.float64)
#     X  = np.zeros((t.shape[0], X0.shape[0]))
#     X[0] = X0
#     hdt = dt*.5
#     for i in range(t.shape[0]-1):
#         t1 = t[i]
#         x1 = X[i]
#         k1 = func(t[i], X[i], m, gamma, P, K, net_addr, net_shape, net_dtype)
        
#         t2 = t[i] + hdt
#         x2 = X[i] + hdt * k1
#         k2 = func(t2, x2, m, gamma, P, K, net_addr, net_shape, net_dtype)
        
#         t3 = t[i] + hdt
#         x3 = X[i] + hdt * k2
#         k3 = func(t3, x3, m, gamma, P, K, net_addr, net_shape, net_dtype)
        
#         t4 = t[i] + dt
#         x4 = X[i] + dt * k3
#         k4 = func(t4, x4, m, gamma, P, K, net_addr, net_shape, net_dtype)
#         X[i+1] = X[i] + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
#     return X

###### RK4 with Numba nopython ######
# @nb.jit(nopython=True)
# def SinIntCover(addr, _shape, _dtype, T):
#     """
#     addr, _shape, _dtype, u[0]
#     """
#     EdgeList = nb.carray(address_as_void_pointer(addr), 
#                          _shape, dtype=_dtype)
#     sinint = SinIntE(EdgeList, T)
#     return sinint


# @nb.jit(nopython=True)

###### swing with lsoda ######
# def swing_lsoda(network, N):
#     """
#     rhs = swing_cover(network)
#     funcptr = rhs.address
#     ...
#     data = np.array([1.0])
#     usol, success = lsoda(funcptr, u0, t_eval, data = data)
#     """
#     @nb.cfunc(lsoda_sig) # network
#     def rhs(t, u, du, p): # p0=m, p1=gamma, p2=P, p3=K
#         u_2D = nb.carray(u, (N,2))
#         Interaction = p[3] * SinIntE(network, u_2D[:,0])
#         du[0] = u_2D[:,1] #Thetas of nodes
#         du[1] = 1/p[0]*(p[2] - p[1]*u_2D[:,1] - Interaction) #Omegas of nodes
        
        
# @cfunc(lsoda_sig)
# def rhs(t, u, du, p):
#     u_2D = nb.carray(u, (1,2))
#     # rest of function goes here
#     du[0] = u_2D[0,0]-u_2D[0,0]*u_2D[0,1]
#     du[1] = u_2D[0,0]*u_2D[0,1]-u_2D[0,1]*p[0]

# funcptr = rhs.address
# u0_2d = np.array([[5.,0.8]])
# u0 = u0_2d.flatten()
# data = np.array([1.0])
# t_eval = np.linspace(0.0,50.0,1000)

# usol, success = lsoda(funcptr, u0, t_eval, data = data)