# swing/initialize.py
"""
Note
-----
Made by Young Jin Kim (kimyoungjin06@gmail.com)
Last Update: 2022.11.09, YJ Kim

Initializing for Modular Structure upto multi-layered Network.
"""

import numpy as np
# import numba as nb
import networkx as nx
import time

# Swing Code 

# from .core import *
from .minor import *

# For Generate Network

def generate_clique(N=3, K=1, init_n=0, _type="core"):
    '''
    Note
    ----
    Generate all-to-all, complete, clique with weight $K$
    '''
    edges = []
    if _type == "core":
        for i in range(init_n, init_n+N):
            for j in range(i+1, init_n+N):
                edges.append([i, j, K])
    return edges

def initialization(N, P, K, gamma=2.0, 
                   t0g=0, w0g=0, t0c=0, w0c=0):
    '''
    Note
    ----
    For Double N-clique Structure
        There is two modular structure.
        One is the generator clique and
        the other is the consumer clique.
        And both side connected by only one bridge edges
        (between bridge nodes.)
    
    Parameter
    ---------
    N : number of nodes in a side.
        Total system have $2N$ nodes.
    P : Power of bridge nodes.
        All other general nodes have unit power.
        $P_{brdg,generator}=P, P_{brdg,consumer}=-P$
        $P_{other,generator}=1, P_{other,consumer}=-1$
    K : Coupling strength between bridge nodes.
    gamma : friction term of the system.
    t0g : init value of $\theta$ (phase) of generator side.
        $\theta_{0,generator}=0$
    w0g : init value of $\omega$ (frequency) of generator side.
        $\omega_{0,generator}=0$
    t0c : init value of $\theta$ (phase) of consumer side.
        $\theta_{0,consumer}=0$
    w0c : init value of $\omega$ (frequency) of consumer side.
        $\omega_{0,consumer}=0$
    
    Return
    ------
    Swing_Parameters, y0, (G, edges_c1, edges_c2, edges_brdg)
        
    ===== example =====
    For detail,
    If N=1, the system is just two nodes.
    If N=2, the system is line segment with 4 nodes.
    If N>2, the system is two-modular structure with
            single bridge edges between two N-cliques.
    
    Backlog
    -------
    - Update to use local gammas
        - Also related with core.swing()
    '''

    # Set the P, M
    P0 = 1
    P1 = +P
    P2 = -P
    Powers = [P0]*(N-1) + [P1] + [P2] + [-P0]*(N-1)

    M1 = 1.
    M2 = 1.
    Inertias = [M1]*N + [M2]*N

    # Set the K
    K1 = 1.
    K2 = 1.
    K_brdg = K

    # 11 type
    edges_c1 = generate_clique(N, K1) # a clique
    edges_c2 = generate_clique(N, K2, N) # another clique
    edges_brdg = [(N-1,N, K_brdg)] # bridge edge

    edges = edges_c1 + edges_c2 + edges_brdg
    
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    adjM = nx.to_numpy_array(G)
    adjL = AdjMtoAdjL(adjM)
    adjE = AdjMtoEdgL(adjM)

    Swing_Parameters = {
        "network": adjE,
        "m": Inertias,
        "gamma": gamma,
        "P": Powers,
        "K": adjM,
        "N": N,
        "model": "swing"
    }
    
    # Initial values
    theta = np.array([t0g]*N+[t0c]*N)
    omega = np.array([w0g]*N+[w0c]*N)
    y0 = np.concatenate(([theta], [omega]), axis=0)
    
    _ = [G, edges_c1, edges_c2, edges_brdg, P0, P1, P2, M1, M2, K1, K2, K_brdg]
    
    return Swing_Parameters, y0, _


def Init_general(N=1024, Backward=False, zero_mean_power=False, degree_type='SF', MAXDegree=0, Lambda=2, esl=1E-4):
    """
    Note
    ----
    - Initialize \theta from 0 to 2\pi (in forward process).
    - Initialize \omega as 0.
    - Initialize power is proportional to degree.
    - Initialize degree as power-law with p(1, MaxDeg, -1*\lambda).
    
    Methods
    -------
    degree, theta, omega, power = Init_Anneal()
    
    Attributes
    ----------
    N : int
        Number of Nodes
    Backward : True or False
        For hysteresis, initialize thetas.
        If False, thetas set to rand(0,1),
        Else if True, thetas set to zero.
    zero_mean_power : True or False
        Condition for Powers have zero-mean.
        If False, $power = degree/<degree>$
    degree_type : 'SF', 'ER', or 'Lattice'
        Degree distribution type of Nodes.
        If 'SF', the degree follows power-law dist
        with MAXDegree and Lambda.
        The esl(epsilon) is very small additional value to Lambda.
        Else if 'ER', the degree follows Poisson dist
        with Lambda.
        Else if 'Lattice', the degree equal to MAXDegree.
    MAXDegree : int
        Max degree (for 'SF' or 'ER') or universal degree (for 'Lattice')
    Lambda : float64
        Parameter of the degree distribution for 'SF' or 'ER'.
    esl : float46
        Add to Lambda for when gamma=1.
    """
    rand = np.random.rand
    tpi = 2*np.pi
    
    if MAXDegree<1:
        try:
            # Lambda = 1
            Lambda += esl
            MAXDegree = int(np.pow(N, 1/(Lambda-1)))
        except:
            MAXDegree = N
    
    theta = tpi*rand(N) # Forward
    if Backward:
        theta = np.zeros(N)
    omega = np.zeros(N)
    if degree_type == 'SF':
        degree = plaw_gen(N, gamma=Lambda, M=MAXDegree).astype(int)
    elif degree_type == 'ER':
        degree = np.random.poisson(Lambda, N).astype(int)
    elif degree_type == 'Lattice':
        degree = (np.ones(N)*MAXDegree).astype(int)
    else:
        raise "Unknown degree type."
        
    deg_mean = degree.mean()
    if zero_mean_power:
        power = (degree - deg_mean)/deg_mean
    else:
        power = (degree)/deg_mean

    return degree, theta, omega, power

# # def set_inits_PK(P, K, t0=0):
# #     '''
# #     For Double 3-core Structure
# #     '''
# #     N = 6
# #     gamma = 0.1
    
# #     # Set the P, M
# #     # P1 = 1.
# #     # P2 = -1.
    
# #     P1 = P
# #     P2 = -P
# #     Powers = [P1, 1, 1, P2, -1, -1]
    
# #     M1 = 1.
# #     M2 = 1.
# #     Inertias = [M1]*3 + [M2]*3
    
# #     # Set the K
# #     K1 = 1.
# #     K2 = 1.
# #     K_brdg = K
    
# #     # 11 type
# #     edges_c1 = [(0,1, K1), (1,2, K1), (2,0, K1)] # a clique
# #     edges_c2 = [(3,4, K2), (4,5, K2), (5,3, K2)] # another clique
# #     edges_brdg = [(0,3, K_brdg)] # bridge edge

# #     edges = edges_c1 + edges_c2 + edges_brdg

# #     # G = nx.from_edgelist(edges)
# #     G = nx.Graph()
# #     G.add_weighted_edges_from(edges)
# #     adjM = nx.to_numpy_array(G)
# #     adjL = AdjMtoAdjL(adjM)
# #     adjE = AdjMtoEdgL(adjM)

# #     Swing_Parameters = {
# #         "network": adjE,
# #         "m": Inertias,
# #         "gamma": gamma,
# #         "P": Powers,
# #         "K": adjM,
# #         "N": N,
# #         "model": "swing"
# #     }
    
# #     theta = np.array([0]*3+[t0]*3)
# #     omega = np.zeros(N) # N=6
# #     y0 = np.concatenate(([theta], [omega]), axis=0)
    
# #     return Swing_Parameters, y0

# def set_inits_PK3(P, K, t0=0):
#     '''
#     For Line Segment
#     '''
#     N = 4
#     gamma = 0.1

#     # Set the P, M
#     # P = 1
#     P1 = +P
#     P2 = -P
#     Powers = [1, P1] + [P2, -1]

#     M1 = 1.
#     M2 = 1.
#     Inertias = [M1]*2 + [M2]*2

#     # Set the K
#     K1 = 1.
#     K2 = 1.
#     K_brdg = K

#     # # 11 type
#     # edges_c1 = [(0,1, K1), (1,2, K1), (2,0, K1)] # a clique
#     # edges_c2 = [(3,4, K2), (4,5, K2), (5,3, K2)] # another clique
#     # edges_brdg = [(0,3, K_brdg)] # bridge edge

#     # 1111 type
#     edges_c1 = [(0,1, K1)] # a clique
#     edges_c2 = [(2,3, K2)] # another clique
#     edges_brdg = [(1,2, K_brdg)] # bridge edge

#     edges = edges_c1 + edges_c2 + edges_brdg

#     # G = nx.from_edgelist(edges)
#     G = nx.Graph()
#     G.add_weighted_edges_from(edges)
#     adjM = nx.to_numpy_array(G)
#     adjL = AdjMtoAdjL(adjM)
#     adjE = AdjMtoEdgL(adjM)

#     Swing_Parameters = {
#         "network": adjE,
#         "m": Inertias,
#         "gamma": gamma,
#         "P": Powers,
#         "K": adjM,
#         "N": N,
#         "model": "swing"
#     }
    
#     theta = np.array([0]*2+[t0]*2)
#     omega = np.zeros(N) # N=6
#     y0 = np.concatenate(([theta], [omega]), axis=0)
    
#     return Swing_Parameters, y0