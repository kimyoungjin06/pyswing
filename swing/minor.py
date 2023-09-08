# swing/minor.py
"""
Note
-----
Made by Young Jin Kim (kimyoungjin06@gmail.com)
Last Update: 2022.11.24, YJ Kim

Some functions related with Network Translation and Phase Diagram
"""

import matplotlib.pyplot as plt
import numpy as np

from . import annealed_multilayer, core

# Related with Network Translation


def AdjMtoAdjL(adjM: np.ndarray) -> list:
    """Adjacency Matrix to Adjacency List"""
    return [np.argwhere(adjM[:, i] > 0).flatten() for i in range(len(adjM))]


def AdjMtoEdgL(adjM: np.ndarray) -> np.ndarray:
    """Adjacency Matrix to Edge List"""
    return np.argwhere(adjM > 0)


# Related with Drawing Phase Diagram


def phase_coherence(angles_vec):
    """
    Compute global order parameter R_t - mean length of resultant vector
    """
    return abs((np.e ** (1j * angles_vec)).mean(axis=1))


def phase_coherence_anneal(angles_vec, degree, meanDegree):
    """
    Compute global order parameter R_t - mean length of resultant vector
    """
    return abs((degree * (np.e ** (1j * angles_vec))).sum(axis=1) / meanDegree)


def mean_frequency(self, act_mat, adj_mat):
    """
    Compute average frequency within the time window (self.T) for all nodes
    """
    assert len(adj_mat) == act_mat.shape[0], "adj_mat does not match act_mat"
    _, n_steps = act_mat.shape

    # Compute derivative for all nodes for all time steps
    dxdt = np.zeros_like(act_mat)
    for time in range(n_steps):
        dxdt[:, time] = self.derivative(act_mat[:, time], None, adj_mat)

    # Integrate all nodes over the time window T
    integral = np.sum(dxdt * self.dt, axis=1)
    # Average across complete time window - mean angular velocity (freq.)
    meanfreq = integral / self.T
    return meanfreq


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
    # Omega Time Mean
    OTM = omega.mean(axis=0)
    OTM_brdg = np.abs(OTM[N - 1: N + 1]).mean()
    OTM_leaf = np.abs([OTM[: N - 1].mean(), OTM[N + 1:].mean()]).mean()
    det = OTM_brdg / OTM_leaf
    return det


# Log-log Fitting
def plfunc(x, g, C, x0, y0):
    """
    note
    ----
    Fitting function with $y = C*(x-x0)^(-G) + y0$.
    """
    return C * (x - x0) ** (-g) + y0


def log_log_fit(func, x, y, **kwargs):
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

    log_x, log_y = np.log(x), np.log(y)

    return scipy.optimize.curve_fit(f_conv, log_x, log_y, **kwargs)


def plot_logfitting(x, nbin=10, label="degree", color="C00"):
    func = plfunc
    xb = np.logspace(np.log10(x.min() - 1), np.log10(x.max() + 1), nbin)
    plt.hist(x, bins=xb, color=color, label=label, density=True)

    hist, bin_edges = np.histogram(x, bins=xb, density=True)
    xbc = xb[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2
    popt, pcov = log_log_fit(func, xbc, hist)
    yfit = func(xbc, *popt)
    g, C, x0, y0 = popt
    # lb = fr"${C:.3f}*(x-{x0:.3f})^{{(-{g:.3f})}} + {y0:.3f}$"
    lb = rf"$y \sim x^{{-{g:.3f}}}$"
    plt.plot(xbc, yfit, "bo-", label=lb)
    plt.xscale("log")
    plt.yscale("log")


def Visualize_inits(inits, degree_type, nbin=20):
    import matplotlib.gridspec as gridspec

    N_ROW = 2
    N_COL = 2
    X_SIZE = 5
    Y_SIZE = 3

    plt.rcParams["font.family"] = ["Helvetica"]
    fig = plt.figure(figsize=(X_SIZE * N_COL, Y_SIZE * N_ROW), dpi=300)
    spec = gridspec.GridSpec(ncols=N_COL, nrows=N_ROW, figure=fig)
    axes = []

    lbs = ["degree", "theta", "omega", "power"]
    for axi, x in enumerate(inits):
        ax = fig.add_subplot(spec[axi // N_COL, axi % N_COL])  # row, col

        if axi > 2:  # power
            plt.hist(x, label=lbs[axi], color=f"C0{axi}", bins=nbin)
            plt.yscale("log")
            # plt.xscale('log')
        elif axi > 0:
            plt.hist(x, label=lbs[axi], color=f"C0{axi}", bins=nbin)
        else:
            if degree_type == "SF":
                plot_logfitting(x, nbin=nbin, label=lbs[axi], color=f"C0{axi}")
            elif degree_type == "Lattice":
                plt.hist(x, label=lbs[axi], color=f"C0{axi}", bins=nbin)
            else:
                plt.hist(
                    x,
                    label=lbs[axi],
                    color=f"C0{axi}",
                    bins=np.arange(x.max()) - 0.5,
                )
        plt.legend()


def Initialize(params, Visualize=True):
    """
     Methods
    -------
    degree, theta, omega, power = Init_Anneal()
    """
    degree_type = params['degree_type']
    
    InitFunc = annealed_multilayer.Init_Anneal
    inits = InitFunc(params)
    if Visualize:
        Visualize_inits(inits, degree_type)
    return inits


def simulate(params, Visualize=True):
    inits = Initialize(params, Visualize=Visualize)
    N, degree_exp, degree_type, MINDegree, MAXDegree, Backward, zero_mean_power, esl = (
        params["N"],
        params["degree_exp"],
        params["degree_type"],
        params["MINDegree"],
        params["MAXDegree"],
        params["Backward"],
        params["zero_mean_power"],
        params["esl"],
    )

    K, M1, COEF_GAMMA, t_end, dt = (
        params["K"],
        params["M1"],
        params["COEF_GAMMA"],
        params["t_end"],
        params["dt"],
    )
    ReduceMemory = params["ReduceMemory"]
    target_time = params["target_time"]
    N_window = int(target_time / dt)

    # Layer1 ordinary
    degree_1, theta_1, omega_1, power_1 = inits
    m_1, gamma_1 = M1, COEF_GAMMA * M1
    totalDegree_1 = degree_1.sum()
    meanDegree_1 = totalDegree_1 / N

    # Total System
    theta, omega = theta_1, omega_1
    X0 = np.concatenate(([theta], [omega]))

    m = np.array([m_1] * N)
    gamma = np.array([gamma_1] * N)
    P = power_1

    degree = degree_1
    totalDegree = totalDegree_1
    meanDegree = meanDegree_1

    kwargs = [totalDegree, meanDegree]
    func = annealed_multilayer.swing_anneal
    if ReduceMemory:
        res = np.zeros((N_window, 2, N))
        t = np.arange(0.0, t_end, dt)
        res[0] = X0
        hdt = dt * 0.5
        for i in range(t.shape[0] - 1):
            res[(i + 1) % N_window] = core.RK4_step(
                func, t[i], res[i%N_window], dt, m, gamma, P, K, degree, *kwargs
            )
    else:
        res = core.RK4(func, t_end, X0, dt, m, gamma, P, K, degree, *kwargs)


    t = np.arange(0, t_end, dt)[-N_window:]
    T, O = res[:, 0, :], res[:, 1, :]
    return t, T, O, P, degree, totalDegree, meanDegree


def draw_snapshot(T, O, degree, meanDegree, params, N_PLOT=5, target_time=1):
    import string

    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    N_PLOT = 5

    N_ROW = N_PLOT
    N_COL = N_PLOT
    X_SIZE = 2
    Y_SIZE = 2
    # plt.rcParams['font.family'] = ['NanumSquare', 'Helvetica']
    plt.rcParams["font.family"] = ["Helvetica"]

    t_end = params["t_end"]
    dt = params["dt"]
    t_e = int(t_end / (N_ROW * N_COL) / dt)
    t = np.arange(0, t_end, dt)
    theta, omega = T.T, O.T

    fig = plt.figure(figsize=(X_SIZE * N_COL, Y_SIZE * (N_ROW + 1)), dpi=300)
    spec = gridspec.GridSpec(
        ncols=N_COL, nrows=N_ROW + 1, figure=fig, wspace=0.4, hspace=0.4
    )  # , width_ratios=[1,1,.1], wspace=.3)
    axes = []
    ts = []
    for axi in range(N_ROW * N_COL):
        ax = fig.add_subplot(spec[axi // N_COL, axi % N_COL])  # row, col
        ax.text(
            0.0,
            1.03,
            rf"$\bf{{{string.ascii_uppercase[axi]}}}$",
            transform=ax.transAxes,
            size=12,
            weight="bold",
        )
        axes.append(ax)

        _t = axi * t_e
        x = theta[:, _t] + 0.5 * np.pi
        _x, _y = np.cos(x), np.sin(x)
        v = omega[:, _t]
        msk = v > 0
        plt.scatter(_x[msk], _y[msk], s=10, alpha=0.1, ec="none", c="red")
        plt.scatter(_x[~msk], _y[~msk], s=10, alpha=0.1, ec="none", c="blue")
        _xm, _ym = np.mean(_x), np.mean(_y)
        plt.plot([0, _xm], [0, _ym], c="orangered", alpha=0.4)
        plt.scatter(_xm, _ym, c="orangered")
        ts.append(_t * dt)
        # Generate circular grid
        for o in np.linspace(0, 2 * np.pi, 13):
            x = [0, np.cos(o)]
            y = [0, np.sin(o)]
            plt.plot(x, y, c="grey", lw=0.3, zorder=-1)

        plt.text(-0.95, 0.95, rf"$t={_t*dt:.1f}$", va="top", ha="left")
        plt.ylim([-1.1, 1.1])
        plt.xlim([-1.1, 1.1])
        plt.xticks(np.linspace(-1.0, 1.0, 3), [f"$-1$", "$0$", f"$1$"])
        plt.yticks(np.linspace(-1.0, 1.0, 3), [f"$-1$", "$0$", f"$1$"])

    axi += 1
    ax = fig.add_subplot(spec[axi // N_COL, :])  # row, col
    ax.text(
        0.0,
        1.03,
        rf"$\bf{{{string.ascii_uppercase[axi]}}}$",
        transform=ax.transAxes,
        size=12,
        weight="bold",
    )
    axes.append(ax)

    end = -1
    # order_param = phase_coherence(T[::])
    # op = phase_coherence_with_LimitCycle(T, degree, meanDegree, params)
    # order_param = swing.minor.phase_coherence_anneal(T, degree, meanDegree)
    totalDegree = int(meanDegree * params['N'])
    new_phase = annealed_multilayer.get_new_phase(T, degree, totalDegree)
    op = annealed_multilayer.get_orderparameter(new_phase, degree, totalDegree)

    plt.plot(t[:end], op[:end], c="royalblue")
    plt.vlines(ts, 0, 1, "orangered")
    plt.hlines(op[-1], 0, t_end, "red", lw=5, zorder=-2)
    for axi, _t in enumerate(ts):
        ax.text(
            _t, 0.1, rf"$\bf{{{string.ascii_uppercase[axi]}}}$", size=12, weight="bold"
        )
    plt.ylim([0, 1])
    plt.xlim([0, t_end])


def phase_coherence_with_LimitCycle(
    phase,
    degree,
    meanDegree,
    params,
    n_circle_LC=1,
    verbose=False,
    Consider_LimitCycle=False,
    Consider_PartiallyPhaseLocking=False,
    Weighted_Average=True,
    Annealed_Transformation=False,
):
    """
    Note
    ----
    - Consider degree-weighted average
    - Limit Cycle의 경우 phase from mean-field가 아닌 실제 phase를 가지고 쟤야 함
        - 적당히 상수를 더해서 shift한 뒤, 마지막 구간만 마스킹해서 각개별 구할 수 있음
        - 추후 필요시 업데이트 예정

    Parameter
    ---------
    phase : the phase of nodes in the system.
    params : Contains all parameter-set.
    N : Number of Nodes in a side

    Return
    ------
    ratio of OTM of bridge and leaf.
    """
    _N = params["N"]
    dt = params["dt"]
    target_time = params["target_time"]
    CONST_dt = params["CONST_dt"]
    totalDegree = meanDegree * _N

    t_unit = int(target_time / dt)
    # print(t_unit)

    # get new phase from mean-field
    if Annealed_Transformation:
        phase = phase[-t_unit:, :]
        phase = phase % (np.pi * 2)
        phase = annealed_multilayer.get_new_phase(phase, degree, totalDegree)

    # masking for too slow for given t_end
    deltaT = phase[-1, :] - phase[-t_unit, :]
    diffT = np.diff(phase[-t_unit:], axis=0)
    move_dist = np.abs(diffT).mean(axis=0)
    msk1 = deltaT > np.pi * 2 * n_circle_LC  # run n_circle_LC circles
    msk2 = diffT.min(axis=0) * diffT.max(axis=0) < 0  # turn direction
    msk3 = move_dist < CONST_dt * dt  # almost stop

    N = 0
    op = 0
    returns = {}
    # Case 1: Limit Cycle
    if Consider_LimitCycle:
        msk = msk1 & ~msk2
        _x, _y = np.cos(phase[-t_unit:, msk]), np.sin(phase[-t_unit:, msk])
        _x_mean, _y_mean = _x.mean(axis=0), _y.mean(axis=0)  # averaging via time-axis

        op1 = (_x_mean**2 + _y_mean**2) ** 0.5
        op1 = op1.sum()  # sum via node axis for op1
        N1 = sum(msk)

        op += op1
        N += N1
        returns["op1"] = op1
        returns["N1"] = N1
        # print(N)

    # Case 2: Partially Phase Locking
    if Consider_PartiallyPhaseLocking:
        msk = ~msk1 & msk2 & ~msk3
        _x, _y = np.cos(phase[-t_unit:, msk]), np.sin(phase[-t_unit:, msk])
        _x_mean, _y_mean = _x.mean(axis=0), _y.mean(axis=0)  # averaging via time-axis

        op2 = (_x_mean**2 + _y_mean**2) ** 0.5
        op2 = op2.sum()
        # op2 = abs((np.e ** (1j * phase[-t_unit:,msk])).sum(axis=1)) # sum via node-axis
        # op2 = op2[-1]
        N2 = sum(msk)

        op += op2
        N += N2
        returns["op2"] = op2
        returns["N2"] = N2
        # print(N)

    # Case 3: Phase Locking
    msk = msk3
    if sum(msk) > 0:
        if Weighted_Average:
            # sum via node-axis
            weighted_sum = abs(
                (degree[msk] * (np.e ** (1j * phase[-t_unit:, msk]))).sum(axis=1)
            )
            # [Q] normed by all node(_N) or summed node (N3)?
            op3 = weighted_sum / degree[msk].mean()  # For summed node (N3)
            # op3 = weighted_sum / meanDegree # For All node (N)
        else:
            op3 = abs((np.e ** (1j * phase[-t_unit:, msk])).sum(axis=1))
        # op3 = op3[-1]
        N3 = sum(msk)
    else:
        op3 = 0.0
        N3 = 0

    op += op3
    N += N3
    returns["op3"] = op3
    returns["N3"] = N3
    # print(N)

    # averaging all order parameters
    if N == 0:
        op = 0.0
    else:
        op = op / N  # [Q]averaging for all node(_N) or summed node (N)?
        op = op.mean()  # time average
    returns["op"] = op
    returns["N"] = N

    if verbose:
        return returns  # , degree[msk], weighted_sum, meanDegree
    else:
        return op