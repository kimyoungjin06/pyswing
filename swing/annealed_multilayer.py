# swing/annealed_multilayer.py
"""
Note
-----
Made by Young Jin Kim (kimyoungjin06@gmail.com)
<<<<<<< HEAD
Update at 2023.05.03, Yj Kim:
- Add 1st-order Kuramoto Model
- Add Blended Model (1st-order with m==0)
- Add weight option in MeanAngFunc
=======
>>>>>>> 0dc8452 (update)

Update at 2023.2.21, YJ Kim:
- Add Regular Sampling with ["Gaussian", "Cauchy", "PowerLaw", "Uniform"]

Update at 2022.12.16, YJ Kim:
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

import time

import networkx as nx
import numba as nb
import numpy as np

from . import core, minor

# Swing Code


# For Core part
# @nb.jit(nopython=True)
# def MeanAngFunc(theta: np.array, degree: np.array,
# totalDegree, func='sine', linearization=False):
#     """
#     Note
#     ----
#     Mean-field trigonometric functions for Scale-Free Network
#     with Annealed Approximation.
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
def MeanAngFunc(
    theta: np.array,
    degree: np.array,
    totalDegree,
    func="sine",
    linearization=False,
<<<<<<< HEAD
    weighted=False
=======
>>>>>>> 0dc8452 (update)
):
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
<<<<<<< HEAD
    if weighted:
        const = degree / totalDegree
    else:
        const = (degree * 0) + (1. / len(theta))
    if func == "cosine":
        MeanValue = (const * np.cos(theta)).sum()
    elif func == "sine":
        if linearization:
            MeanValue = (const * theta).sum()
        else:
            MeanValue = (const * np.sin(theta)).sum()
=======
    if func == "cosine":
        MeanValue = (degree * np.cos(theta)).sum() / totalDegree
    elif func == "sine":
        MeanValue = (degree * np.sin(theta)).sum() / totalDegree
        if linearization:
            MeanValue = (degree * theta).sum() / totalDegree
>>>>>>> 0dc8452 (update)
    # else:
    #     raise "Unknown function type; Only 'sine' or 'cosine'"
    return MeanValue


@nb.jit(nopython=True)
def MeanInt(theta1, theta2, degree, totalDegree, meanDegree):
    """
    Note
    ----

    """

    MS = MeanAngFunc(theta2, degree, totalDegree, func="sine")
    MC = MeanAngFunc(theta2, degree, totalDegree, func="cosine")
    Interactions = (
        MS * np.cos(theta1) - MC * np.sin(theta1)
    ) * degree #/ degree.shape[-1] / meanDegree
    return Interactions


def swing_anneal(
    t, y, m, gamma, P, K, degree, totalDegree, meanDegree
) -> np.array([[]]):
    """
    \dot{\theta} &= \omega \\
    \dot{\omega} &= \frac{1}{m}(P-\gamma\omega+\Sigma K\sin(\theta-\phi))
    """
    T, O = y

    # Get Interaction
    #     Interaction = K*SinIntCover(net_addr, net_shape, net_dtype, T)
    Interaction = K * MeanInt(T, T, degree, totalDegree, meanDegree)
    dT = O
    dO = 1 / m * (P - gamma * O + Interaction)
    dydt = np.concatenate(([dT], [dO]))  # , dtype=np.float64)
    return dydt


<<<<<<< HEAD
def Kuramoto(t, y, m, gamma, P, K, degree, totalDegree, meanDegree) -> np.array([[]]):
    """
    \dot{\theta} &= \omega \\
    \dot{\omega} &= \frac{1}{m}(P-\gamma\omega+\Sigma K\sin(\theta-\phi))
    """
    T = y

    m = np.array(m)
    P = np.array(P)

    # Get Interaction
#     Interaction = K*SinIntCover(net_addr, net_shape, net_dtype, T)
    Interaction =  K * MeanInt(T, T, degree, totalDegree, meanDegree)
    dT = (P + Interaction)
    dydt = dT#, dtype=np.float64)
    return dydt


def Blended(t, y, m, gamma, P, K, degree, totalDegree, meanDegree) -> np.array([[]]):
    """
    Some nodes (without mass) interact like 1st order Kuramoto model.
    Other nodes (with mass) interact like 2nd order Kuramoto model.
    """
    T, O = y

    m = np.array(m)
    P = np.array(P)

    # Get Interaction
#     Interaction = K*SinIntCover(net_addr, net_shape, net_dtype, T)
    Interaction =  K * MeanInt(T, T, degree, totalDegree, meanDegree)

    # Masking
    msk = m > 0
    msk_1st = ~msk
    msk_2nd = msk

    dT = np.zeros(m.shape[0])
    dO = np.zeros(m.shape[0])

    ## 1st
    dT[msk_1st] = (P[msk_1st] + Interaction[msk_1st])

    ## 2nd
    dT[msk_2nd] = O[msk_2nd]
    dO[msk_2nd] = 1/m[msk_2nd]*(P[msk_2nd] - gamma[msk_2nd]*O[msk_2nd] + Interaction[msk_2nd])
    dydt = np.concatenate(([dT], [dO]))#, dtype=np.float64)
    return dydt


=======
>>>>>>> 0dc8452 (update)
def swing_anneal_twoLayer(
    t, y, m, gamma, P, K, degree, totalDegree, meanDegree
) -> np.array([[]]):
    """
    two network with intra edges
    \dot{\theta} &= \omega \\
    \dot{\omega} &= \frac{1}{m}(P-\gamma\omega+\Sigma K\sin(\theta-\phi))
    """
    T, O = y
    N = len(T) // 2

    T_1, T_2 = T[:N], T[N:]
    O_1, O_2 = O[:N], O[N:]
    m_1, m_2 = m[:N], m[N:]
    gamma_1, gamma_2 = gamma[:N], gamma[N:]
    P_1, P_2 = P[:N], P[N:]
    degree_1, degree_2, degree_I1, degree_I2 = (
        degree[:N],
        degree[N : N * 2],
        degree[N * 2 : N * 3],
        degree[N * 3 :],
    )
    totalDegree_1, totalDegree_2, totalDegree_I1, totalDegree_I2 = totalDegree
    meanDegree_1, meanDegree_2, meanDegree_I1, meanDegree_I2 = meanDegree

    # Get Interaction
    # Layer 1
    Interaction_1 = K * MeanInt(T_1, T_1, degree_1, totalDegree_1, meanDegree_1)
    Interaction_1I = K * MeanInt(
        T_2, T_1, degree_I2, totalDegree_I2, meanDegree_I2
    )
    dT_1 = O_1
    dO_1 = 1 / m_1 * (P_1 - gamma_1 * O_1 + Interaction_1 + Interaction_1I)

    # Layer 1
    Interaction_2 = K * MeanInt(T_2, T_2, degree_2, totalDegree_2, meanDegree_2)
    Interaction_2I = K * MeanInt(
        T_1, T_2, degree_I1, totalDegree_I1, meanDegree_I1
    )
    dT_2 = O_2
    dO_2 = 1 / m_2 * (P_2 - gamma_2 * O_2 + Interaction_2 + Interaction_2I)

    dT = np.append(dT_1, dT_2)
    dO = np.append(dO_1, dO_2)
    dydt = np.concatenate(([dT], [dO]))  # , dtype=np.float64)
    return dydt


# For Initialization
def Init_Anneal(
    params
):
    """
    Note
    ----
    - Initialize \theta from 0 to 2\pi (in forward process).
    - Initialize \omega as 0.
    - Initialize power is proportional to degree.
    - Initialize degree as power-law with p(1, MaxDeg, -1*\degree_exp).

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
    degree_type : 'SF', 'ER', 'Lattice', or 'FC'
        Degree distribution type of Nodes.
        If 'SF', the degree follows power-law dist
        with MAXDegree and degree_exp.
        The esl(epsilon) is very small additional value to degree_exp.
        Else if 'ER', the degree follows Poisson dist
        with Lambda(degree_exp).
        Else if 'Lattice', the degree equal to MAXDegree.
        Else if 'FC', Fully Connected; All to All. It is sfecific type of Lattice.
    degree_exp : float64
        Parameter of the degree distribution for 'SF', 'ER', or 'Gaussian'.
    MAXDegree : int
        Max degree (for 'SF' or 'ER') or universal degree (for 'Lattice')
    power_type : 'Gaussian', 'Cauchy', 'PowerLaw', or 'Uniform'

    power_exp : float64
        If 'Gaussian', it is standard deviation(Sigma) of Gaussian.
        Else if 'Cauchy', it is the gamma of Cauchy.
        Else if 'PowerLaw', it is the gamma of Power Law.

    esl : float46
        Add to degree_exp for when gamma=1.
    """
    N = params['N']
    Backward = params['Backward']
    degree_type = params['degree_type']
    power_type = params['power_type']
    RegularSampling = params['RegularSampling']
<<<<<<< HEAD
    Irregular_Theta = params["Irregular_Theta"]

    theta = np.pi + Uniform_distribution(np.pi, N, RegularSampling)  # Forward
    if Irregular_Theta:
        theta = Uniform_distribution(np.pi, N, False)
=======

    theta = np.pi + Uniform_distribution(np.pi, N, RegularSampling)  # Forward
>>>>>>> 0dc8452 (update)
    if Backward:
        theta = np.zeros(N)
    omega = np.zeros(N)

    # Initialize Degree distribution
    if degree_type == "SF":
        degree_exp = params['degree_exp']
        MINDegree = params['MINDegree']
        MAXDegree = params['MAXDegree']
        esl = params['esl']
        degree = PowerLaw_distribution(
            degree_exp, N, RegularSampling, MIN=MINDegree, MAX=MAXDegree, esl=esl
        ).astype(int)

    elif degree_type == "ER":
        degree_exp = params['degree_exp']
        degree = np.random.poisson(degree_exp, N).astype(int)
        degree = np.maximum(
            degree, 1.0
        )  # For degree = 0 with low Lambda, d=0 > d=1
    elif degree_type == "Lattice":
        MAXDegree = params['MAXDegree']
        if MAXDegree == 0:
            MAXDegree = 4
        degree = (np.ones(N) * MAXDegree).astype(int)
    elif degree_type == "FC":
        degree = np.ones(N).astype(int)
    else:
        raise Exception("Unknown degree type.")
    deg_mean = degree.mean()

    # Initialize Power
    if power_type == "Gaussian":
        power_exp = params['power_exp']
        power = Gaussian_distribution(power_exp, N, RegularSampling)
    elif power_type == "Cauchy":
        power_exp = params['power_exp']
        power = Cauchy_distribution(power_exp, N, RegularSampling)
    elif power_type == "PowerLaw":
        zero_mean_power = params['zero_mean_power']
        MINPower = params['MINPower']
        MAXPower = params['MAXPower']
        esl = params['esl']
        if zero_mean_power:
            power = (degree - deg_mean) / deg_mean
        else:
            power = PowerLaw_distribution(
                power_exp, N, RegularSampling, MIN=MINPower, MAX=MAXPower, esl=esl
            )
    elif power_type == "Uniform":
        power_exp = params['power_exp']
        power = Uniform_distribution(power_exp, N, RegularSampling)
    else:
        raise Exception("Unknown power type.")

    return degree, theta, omega, power


def Gaussian_distribution(sigma, N, RegularSampling):
    """
    note
    ----
    Sampling from Gaussian(Normal) distribution.
    """
    import scipy

    if RegularSampling:
        x = np.arange(1, N + 1)
        np.random.shuffle(x)
        return np.sqrt(2) * sigma * scipy.special.erfinv(-1 + (2 * x - 1) / N)
    else:
        return np.random.normal(0, sigma, N)


def Cauchy_distribution(gamma, N, RegularSampling):
    """
    note
    ----
    Sampling from Cauchy distribution (Lorentzian).
    We assume the x_0 is 0.
    """
    if RegularSampling:
        x = np.arange(1, N + 1)
        np.random.shuffle(x)
        return gamma * np.tan(np.pi * 0.5 * (-1 + (2 * x) / (N + 1)))
    else:
        x = np.random.rand(N)
        return gamma * np.tan(np.pi * (x - 0.5))


def PowerLaw_distribution(
    gamma,
    N,
    RegularSampling,
    MIN=1,
    MAX=0,
    esl=1e-2
):
    """
    note
    ----
    Sampling from Power-law distribution propto x^{-(gamma-1)} for m<=x<=M
    with 2 < gamma < 3.
    If Regular Sampling, p(x) = ((Lambda-1)/x_min) * (x/x_min)^(-Lambda).
    """
    if RegularSampling:
        x = np.arange(1, N + 1)[::-1]
        np.random.shuffle(x)
        if MAX <= MIN:  # No Upper bound Case
            # MAX = int(np.power(N, 1 / (gamma - 1)))
            return MIN * (N / x) ** (1 / (gamma - 1))
        else:  # with Upper bound Case
            mgp1 = -gamma + 1
            km, kM = MIN**mgp1, MAX**mgp1
            return (kM + (x/N)*(km - kM))**(1/mgp1)
    else:
        if MAX < 1:
            try:
                gamma += esl
                MAX = int(np.power(N, 1 / (gamma - 1)))
            except:
                MAX = N

        gamma -= 1  # Because of {gamma-1}
        gamma *= -1
        x = np.random.random(size=N)
        mg, Mg = MIN**gamma, MAX**gamma
        return (mg + (Mg - mg) * x) ** (1.0 / gamma)


def Uniform_distribution(gamma, N, RegularSampling):
    """
    note
    ----
    Sampling from Uniform distribution in range [-gamma,gamma].
    """
    if RegularSampling:
        x = np.arange(1, N + 1)
        np.random.shuffle(x)
        return gamma * (-1 + (2 * x - 1) / N)
    else:
        return np.random.uniform(-gamma, gamma, N)



# def single_layer_initialize(Initialize, params):
#     # For Inits
#     N = params["N"]
#     degree_type = params["degree_type"]  # or SF, ER
#     degree_exp = params["degree_exp"] # For SF, 1 < Lambda < 3
#     MINDegree = params["MINDegree"]  # For Lattice, MAXDegree != 0
#     MAXDegree = params["MAXDegree"]  # For Lattice, MAXDegree != 0
#     power_type = params["power_type"]
#     power_exp = params["power_exp"]
#     MINPower = params["MINPower"]
#     MAXPower = params["MAXPower"]
#     RegularSampling = params["RegularSampling"]
#     Backward = params["Backward"]
#     zero_mean_power = params["zero_mean_power"]
#     esl = params["esl"]  # For SF, Add to Lambda for when gamma=1

#     inits = Initialize(
#         N=N,
#         Backward=Backward,
#         zero_mean_power=zero_mean_power,
#         degree_type=degree_type,
#         degree_exp=degree_exp,
#         MINDegree=MINDegree,
#         MAXDegree=MAXDegree,
#         power_type=power_type,
#         power_exp=power_exp,
#         MINPower=MINPower,
#         MAXPower=MAXPower,
#         RegularSampling=RegularSampling,
#         esl=esl,
#     )
#     return inits


def get_layer_information(inits, M, COEF_GAMMA, N):
    """
    Note
    ----
    Extract Layer Information for swing with init values, M, and COEF_GAMMA.
    **Anealed Version**
    - gamma is propotional with M.

    Parameter
    ---------
    inits : list
        [degree, theta, omega, power]
    M : float
        Contains all parameter-set.
    COEF_GAMMA : float
        Coefficient for gamma from M.
    """
    degree, theta, omega, power = inits
    m, gamma = M, COEF_GAMMA * M
    totalDegree = degree.sum()
    meanDegree = totalDegree / N

    # # Layer1 ordinary
    # degree_1, theta_1, omega_1, power_1 = inits
    # m_1, gamma_1 = M1, COEF_GAMMA * M1
    # totalDegree_1 = degree_1.sum()
    # meanDegree_1 = totalDegree_1 / N

    #     # Total System
    #     theta, omega = theta_1, omega_1
    #     X0 = np.concatenate(([theta], [omega]))

    #     m = np.array([m_1]*N)
    #     gamma = np.array([gamma_1]*N)
    #     P = power_1
    #     K = K

    #     degree = degree_1
    #     totalDegree = totalDegree_1
    #     meanDegree = meanDegree_1
    return m, gamma, power, degree, totalDegree, meanDegree


def single_layer_configuration(inits, params):
    """
    Note
    ----
    Configuration for swing with init values and parameters.

    Parameter
    ---------
    inits : list
        [degree, theta, omega, power]
    params : dict
        Contains all parameter-set.
    """
    N = params["N"]
    K = params["K"]
    M1 = params["M1"]
    COEF_GAMMA = params["COEF_GAMMA"]
    m, gamma, P, degree, totalDegree, meanDegree = get_layer_information(
        inits, M1, COEF_GAMMA, N
    )
    return m, gamma, P, K, degree, totalDegree, meanDegree


# Get OrderParameter
def solve_func_with_orderparam(params):
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
    t_end = params["t_end"]
    dt = params["dt"]
    target_time = params["target_time"]

    # Configuration
    Initialize = Init_Anneal
    inits = Initialize(params)
    (
        m,
        gamma,
        P,
        K,
        degree,
        totalDegree,
        meanDegree,
    ) = single_layer_configuration(inits, params)

    X0 = np.concatenate(([inits[1]], [inits[2]]))
    kwargs = [totalDegree, meanDegree]
    func = swing_anneal
    res = core.RK4(func, t_end, X0, dt, m, gamma, P, K, degree, *kwargs)

    T, O = res[:, 0, :], res[:, 1, :]
    duration = int(5.0 / dt)

    order_param = minor.phase_coherence_with_LimitCycle(
        T, degree, meanDegree, params
    )
    return order_param


def solve_func_with_orderparam2(params):
    """
    Note
    ----
    From solve_func_with_orderparam(),
    Change the get order parameter, with new phase.

    Parameter
    ---------
    params : dict
        Contains all parameter-set.
    """

    # Extract from params
    N = params["N"]
    ReduceMemory = params["ReduceMemory"]
    t_end = params["t_end"]
    dt = params["dt"]
    target_time = params["target_time"]
    N_window = int(target_time / dt)

    # Configuration
    Initialize = Init_Anneal
    inits = Initialize(params)
    (
        m,
        gamma,
        P,
        K,
        degree,
        totalDegree,
        meanDegree,
    ) = single_layer_configuration(inits, params)

    X0 = np.concatenate(([inits[1]], [inits[2]]))
    kwargs = [totalDegree, meanDegree]
    func = swing_anneal
    if ReduceMemory:
        res = np.zeros((N_window, 2, N))
        t = np.arange(0.0, t_end, dt)
        res[0] = X0
        hdt = dt * 0.5
        for i in range(t.shape[0] - 1):
            res[(i + 1) % N_window] = core.RK4_step(
                func, t[i], res[i%N_window], dt, m, gamma, P, K, degree, *kwargs
            )
        T, O = res[:, 0, :], res[:, 1, :]
    else:
        res = core.RK4(func, t_end, X0, dt, m, gamma, P, K, degree, *kwargs)
        T, O = res[N_window:, 0, :], res[N_window:, 1, :]

    sin = np.sin(T)
    cos = np.cos(T)
    phase_new, phase_mean = get_new_phase(T, degree, totalDegree, sin=sin, cos=cos)
    sin = np.sin(phase_new)
    cos = np.cos(phase_new)
    order_param = get_orderparameter(phase_new, degree, totalDegree, N=N_window, sin=sin, cos=cos)
    # order_param = get_orderparameter(
    #     T, degree, totalDegree, N=N_window
    # )
    return order_param[-1]


def get_order_param(params):
    from tqdm import tqdm

    K_stt = params["K_stt"]
    K_end = params["K_end"]
    dK = params["dK"]
    Ensemble = params["Ensemble"]
    K_space = np.arange(K_stt, K_end, dK)
    rs = []
    for K in tqdm(K_space):
        params["K"] = K
        _rs = []
        for Ei in range(Ensemble):
            res = solve_func_with_orderparam(params)
            _rs.append(res)
        rs.append(_rs)
    return K_space, np.array(rs)


# int main():
#     gamma, Delta, dt, tEnd, N,
#     CouplingConst, Force, MeanCosTheta, MeanSinTheta, deltaT
#     theta, omega, degree, power = Init_Aneal(N, gamma, , esl=1E-4)


# Minor #####################
def get_new_phase(phase, degree, totalDegree, sin=None, cos=None):
    """
    Note
    ----
    Get new phase from mean-phase
    Pre calculate Sin and Cosine
    """
    if sin is None:
        sin = np.sin(phase)
        cos = np.cos(phase)
    MS = (degree * sin).sum(axis=1) / totalDegree
    MC = (degree * cos).sum(axis=1) / totalDegree
    phase_mean = np.arctan2(MS, MC)
    phase_new = phase - phase_mean.reshape(-1, 1)
    return phase_new, phase_mean


def get_orderparameter(
    phase, degree, totalDegree, TimeAverageFirst=True, N=1000, sin=None, cos=None
):
    """
    Note
    ----
    Get order parameter.
    If TimeAverageFirst, apply moving average for all nodes.
    N is the window size for dt-steps.
    """
    if sin is None:
        sin = np.sin(phase)
        cos = np.cos(phase)
    if TimeAverageFirst:
        sin = time_average(sin, N)
        cos = time_average(cos, N)
    MS = (degree * sin).sum(axis=1) / totalDegree
    MC = (degree * cos).sum(axis=1) / totalDegree

    r = np.sqrt(MS**2 + MC**2)
    return r


def time_average(x, N):
    x = np.insert(x, 0, x[0], 0)
    cumsum = x.cumsum(axis=0)
    return (cumsum[N:] - cumsum[:-N]) / N