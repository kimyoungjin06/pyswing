# swing/annealed.py
"""
Note
-----
Made by Young Jin Kim (kimyoungjin06@gmail.com)
Last Update: 2022.12.16, YJ Kim

For annealed version of swing equation on Scale-Free Network.

Example
-------
- Two mass system

import pyswing.swing as swing

N = 1000
Lambda = 2
lbs = ['theta', 'omega', 'power', 'degree']
inits = swing.annealed.Init_Anneal(N, Lambda, esl=1E-4, zero_mean=False)
# inits = Init_Anneal(N, Lambda, esl=1E-4)

t_end = 100.
dt = 0.01
m, gamma, P, K = 1., 20., power, KK
m_space = np.array([0.1, MM])
# m = m_space[np.random.randint(0, len(m_space), N)] # Randomly setting
m = m_space[(degree//degree.mean()>0).astype(int)] # Proportional with degree

totalDegree = degree.sum()
meanDegree = totalDegree / N

X0 = np.concatenate(([theta], [omega]))
kwargs = [totalDegree, meanDegree]
res = swing.core.RK4(swing.annealed.swing_anneal, t_end, X0, dt, m, gamma, P, K, degree, *kwargs)

# Plotting
N_ROW = 1
N_COL = 1
X_SIZE = 12
Y_SIZE = 6
fig=plt.figure(figsize = (X_SIZE*N_COL,Y_SIZE*N_ROW), dpi=300)

t_e = int(t_end/dt)-1
t = np.arange(0, t_end, dt)

T, O = res[:,0,:], res[:,1,:]
dgrs = np.logspace(0, np.log10(degree.max()), 10)
dashes = [(4, 1), (1, 0)]
colors = ['blue', 'red']
width = [.5, .5]
for i, dgr in enumerate(dgrs[:-1]):
    for j, mm in enumerate(m_space):
        msk = (dgr <= degree) & (degree < dgrs[i+1]+1)
        msk2 = m == mm
        _ = plt.plot(t[:t_e], O[:t_e,msk & msk2], c=f"C{i}", alpha=.2, dashes=dashes[j], lw=width[j])
        _ = plt.plot(t[:t_e], O[:t_e,msk & msk2].mean(axis=1), c=f"C{i}", dashes=dashes[j], lw=width[j])
plt.xlabel('t')
_ = plt.ylabel(fr'$\omega$')
_ = plt.xlim([0, t[t_e]])

"""

import numpy as np
import numba as nb
import networkx as nx
import time

# Swing Code 

from .core import *
# from .minor import *


## For Core part
# @nb.jit(nopython=True)
# def MeanAngFunc(theta: np.array, degree: np.array, totalDegree, func='sine', linearization=False):
#     """
#     Note
#     ----
#     Mean-field trigonometric functions for Scale-Free Network with Annealed Approximation.
#     - Added linearization for when func is 'sine'
    
#      Attributes
#     ----------
#     func : 'sine' or 'cosine'
#         A formatted string to print out what the animal says
#     """
#     if func == 'cosine':
#         func_ = np.cos
#     elif func == 'sine':
#         func_ = np.sin
#         if linearization:
#             func_ = lambda x:x
#     # else:
#     #     raise "Unknown function type; Only 'sine' or 'cosine'"

#     MeanValue = float(degree * func_(theta)).sum()/totalDegree
#     return MeanValue

@nb.jit(nopython=True)
def MeanAngFunc(theta: np.array, degree: np.array, totalDegree, func='sine', linearization=False):
    """
    Note
    ----
    Mean-field trigonometric functions for Scale-Free Network with Annealed Approximation.
    - Added linearization for when func is 'sine'
    
     Attributes
    ----------
    func : 'sine' or 'cosine'
        A formatted string to print out what the animal says
    """
    if func == 'cosine':
        MeanValue = (degree * np.cos(theta)).sum()/totalDegree
    elif func == 'sine':
        MeanValue = (degree * np.sin(theta)).sum()/totalDegree
        if linearization:
            MeanValue = (degree * theta).sum()/totalDegree
    # else:
    #     raise "Unknown function type; Only 'sine' or 'cosine'"
    return MeanValue

@nb.jit(nopython=True)
def MeanInt(theta, degree, totalDegree, meanDegree):
    """
    Note
    ----
    Mean-field Interactions for Scale-Free Network with Annealed Approxization.
    
    """

    MS = MeanAngFunc(theta, degree, totalDegree, func="sine")
    MC = MeanAngFunc(theta, degree, totalDegree, func="cosine")
    Interactions = (MS*np.cos(theta) - MC*np.sin(theta)) * degree / meanDegree
    return Interactions

def swing_anneal(t, y, m, gamma, P, K, degree, totalDegree, meanDegree) -> np.array([[]]):
    """
    \dot{\theta} &= \omega \\
    \dot{\omega} &= \frac{1}{m}(P-\gamma\omega+\Sigma K\sin(\theta-\phi))
    """
    T, O = y
    
    # Get Interaction
#     Interaction = K*SinIntCover(net_addr, net_shape, net_dtype, T)
    Interaction = K*MeanInt(T, degree, totalDegree, meanDegree)
    dT = O
    dO = 1/m*(P - gamma*O + Interaction)
    dydt = np.concatenate(([dT], [dO]))#, dtype=np.float64)
    return dydt

## For Initialization
def Init_Anneal(N=1024, Backward=False, zero_mean_power=False, degree_type='SF', MAXDegree=0, Lambda=2, esl=1E-4):
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



# int main():
#     gamma, Delta, dt, tEnd, N, 
#     CouplingConst, Force, MeanCosTheta, MeanSinTheta, deltaT
#     theta, omega, degree, power = Init_Aneal(N, gamma, , esl=1E-4)



################ Minor #####################

def plaw_gen(N, gamma=3, m=1, M=2, ):
    """
    note
    ----
    Power-law gen for pdf(x)\propto x^{-(gamma-1)} for m<=x<=M"""
    gamma -= 1 # Because of {gamma-1}
    gamma *= -1
    x = np.random.random(size=N)
    mg, Mg = m**gamma, M**gamma
    return (mg + (Mg - mg)*x)**(1./gamma)
