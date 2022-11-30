import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import font_manager
import string

from .core import *
from .initialize import *
from .multiprocessing import *
from .minor import *

font_manager.fontManager.addfont('/disk/disk2/youngjin/workspace/Helvetica.ttf')

def get_result(swing_param, visualize_network=False):
    N, P, K, gamma, t_end, dt, t0g, w0g, t0c, w0c = swing_param
    NN = N*2

    Swing_Parameters, y0, _ = initialization(N, P, K, gamma, t0g, w0g, t0c, w0c)
    
    if visualize_network:
        draw_network(N, _)

    params = {}
    params['sparam'] = Swing_Parameters
    params['t_end'] = t_end
    params['init'] = y0
    params['dt'] = dt
    res = solve_func(params)


    P_grid, K_grid, g_grid = 1, 1, 1

    t_eval = np.arange(0,t_end, dt)
    res = np.array(res)
    res2 = res.reshape(P_grid, K_grid, g_grid, t_eval.shape[0], 2, NN) # 2 means theta, omega


    _P = 0
    _K = 0
    _g = 0
    theta = res2[_P, _K, _g][:,0,:] # t, 0, node_idx
    omega = res2[_P, _K, _g][:,1,:] # t, 1, node_idx
    
    return theta, omega

def get_result_with_determider(swing_param, visualize_network=False):
    N, P, K, gamma, t_end, dt, t0g, w0g, t0c, w0c = swing_param
    NN = N*2

    Swing_Parameters, y0, _ = initialization(N, P, K, gamma, t0g, w0g, t0c, w0c)
    
    if visualize_network:
        draw_network(N, _)

    params = {}
    params['sparam'] = Swing_Parameters
    params['t_end'] = t_end
    params['init'] = y0
    params['dt'] = dt
    res = solve_func(params)


    P_grid, K_grid, g_grid = 1, 1, 1

    t_eval = np.arange(0,t_end, dt)
    res = np.array(res)
    res2 = res.reshape(P_grid, K_grid, g_grid, t_eval.shape[0], 2, NN) # 2 means theta, omega


    _P = 0
    _K = 0
    _g = 0
    theta = res2[_P, _K, _g][:,0,:] # t, 0, node_idx
    omega = res2[_P, _K, _g][:,1,:] # t, 1, node_idx
    det = determiner(omega, N)
    return det


def draw_network(N, multi_param):
    G, edges_c1, edges_c2, edges_brdg, P0, P1, P2, M1, M2, K1, K2, K_brdg = multi_param
    pos = nx.spring_layout(G)
    
    N_ROW = 1
    N_COL = 1
    X_SIZE = 10
    Y_SIZE = 10
    # plt.rcParams['font.family'] = ['NanumSquare', 'Helvetica']
    plt.rcParams['font.family'] = ['Helvetica']
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    fig=plt.figure(figsize = (X_SIZE*N_COL,Y_SIZE*N_ROW), dpi=300)

    # nodes
    options = {"edgecolors": "tab:gray", "node_size": 300, "alpha": 0.6}
    nx.draw_networkx_nodes(G, pos, nodelist=np.arange(N-1), node_color="tab:red", **options)
    nx.draw_networkx_nodes(G, pos, nodelist=np.arange(N+1,N*2), node_color="tab:blue", **options)
    options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 0.9}
    nx.draw_networkx_nodes(G, pos, nodelist=np.arange(N-1, N), node_color="tab:red", **options)
    nx.draw_networkx_nodes(G, pos, nodelist=np.arange(N, N+1), node_color="tab:blue", **options)

    # edges
    # nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges_c1,
        width=8,
        alpha=0.5,
        edge_color="tab:red",
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges_c2,
        width=8,
        alpha=0.5,
        edge_color="tab:blue",
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges_brdg,
        width=8,
        alpha=0.5,
        edge_color="black",
    )

    # some math labels
    # labels = {}
    # labels[0] = r"$b$"
    # labels[1] = r"$a$"
    # labels[2] = r"$\alpha$"
    # labels[3] = r"$\beta$"

    # _ = nx.draw_networkx_labels(G, pos, labels, font_size=22, font_color="whitesmoke")

    # edge_labels = nx.get_edge_attributes(G, "weight")
    # nx.draw_networkx_edge_labels(G, pos, edge_labels)

    _ = plt.axis("off")

    _ = plt.scatter([],[], c="tab:red", ec="tab:gray", s=300, alpha=.6, label=fr"$P_{{1,leaf}}={P0}, M_1={M1}$")
    _ = plt.scatter([],[], c="tab:blue", ec="tab:gray", s=300, alpha=.6, label=fr"$P_{{2,leaf}}={-P0}, M_2={M2}$")
    _ = plt.scatter([],[], c="tab:red", ec="tab:gray", s=800, alpha=.9, label=fr"$P_{{1,Joint}}={P1}, M_1={M1}$")
    _ = plt.scatter([],[], c="tab:blue", ec="tab:gray", s=800, alpha=.9, label=fr"$P_{{2,Joint}}={P2}, M_2={M2}$")
    _ = plt.plot([],[], c="tab:red", lw=8, alpha=.5, label=fr"$K_1={K1}$")
    _ = plt.plot([],[], c="tab:blue", lw=8, alpha=.5, label=fr"$K_2={K2}$")
    _ = plt.plot([],[], c="black", lw=8, alpha=.5, label=fr"$K_{{brdg}}={K_brdg}$")
    plt.legend(bbox_to_anchor=[1,1], fontsize=20)
    
    
# Drawing Part
    
def draw_figure(theta, omega, swing_param):
    N, P, K, gamma, t_end, dt, t0g, w0g, t0c, w0c = swing_param
    labels = ["leaf"] + ["Joint"] + ["Joint"] + ["leaf"]

    N_ROW = 4
    N_COL = 1
    X_SIZE = 3
    Y_SIZE = 1.2
    # plt.rcParams['font.family'] = ['NanumSquare', 'Helvetica']
    plt.rcParams["font.family"] = ["Helvetica"]
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Helvetica"
    })
    legend_fontsize=5

    fig=plt.figure(figsize = (X_SIZE*N_COL,Y_SIZE*N_ROW), dpi=200)
    spec = gridspec.GridSpec(ncols=N_COL, nrows=N_ROW, figure=fig, wspace=.4, hspace=.8)#, width_ratios=[1,1,.1], wspace=.3)
    axes = []

    # theta = res2[ENS,ENS2][:,0,:]
    # omega = res2[ENS,ENS2][:,1,:]
    # theta = center_mod(theta)

    v = omega[1,:]
    msk = np.array([True]*(N) + [False]*(N))
    t_eval = np.arange(0,t_end, dt)

    axi=0
    ax = fig.add_subplot(spec[axi//N_COL,axi%N_COL]) # row, col
    ax.text(.0, 1.03, fr"$\bf{{{string.ascii_uppercase[axi]}}}$", transform=ax.transAxes, size=12)
    axes.append(ax)
    plt.plot(t_eval, abs((np.e**(1j*theta[:, msk])).mean(axis=1)), color='red', alpha=.5, lw=.5, label=r'$\omega>0$')
    plt.plot(t_eval, abs((np.e**(1j*theta[:, ~msk])).mean(axis=1)), color='blue', alpha=.5, lw=.5, label=r'$\omega><$')
    plt.plot(t_eval, abs((np.e**(1j*theta[:,:])).mean(axis=1)), color='black', alpha=.5, lw=.5, label=r'total')
    plt.ylabel(r'$r$')
    # plt.title(f"{PK[ENS,ENS2][0]:.1f}, {PK[ENS,ENS2][1]:.1f}")
    plt.title(fr"$P={P}, K={K}, \gamma={gamma}$")


    axi=1
    ax = fig.add_subplot(spec[axi//N_COL,axi%N_COL]) # row, col
    # ax.text(.0, 1.03, fr"$\bf{string.ascii_uppercase[axi]}$", transform=ax.transAxes, size=12, weight="bold")
    axes.append(ax)
    plt.plot(t_eval, omega[:, msk].mean(axis=1), color='red', alpha=.5, label=r'$\omega>0$')
    plt.plot(t_eval, -omega[:, ~msk].mean(axis=1), color='blue', alpha=.5, label=r'$\omega<0$')
    plt.plot(t_eval, -omega[:,:].mean(axis=1), color='black', alpha=.5, label=r'total')
    plt.ylabel(r'$\bar{\omega}$')
    plt.xlabel(r'$t$')
    plt.legend(fontsize=legend_fontsize)

    axi=2
    ax = fig.add_subplot(spec[axi//N_COL,axi%N_COL]) # row, col
    ax.text(.0, 1.03, fr"$\bf{{{string.ascii_uppercase[axi]}}}$", transform=ax.transAxes, size=12, weight="bold")
    axes.append(ax)

    dashes = [(4., 4.), (2., .6)]
    for i in range(N-1):
        plt.plot(t_eval, theta[:,i], dashes=dashes[0], lw=1,
                color="blue", alpha=.1)
    for i in range(N+1,N*2):
        plt.plot(t_eval, theta[:,i], dashes=dashes[0], lw=1,
                color="red", alpha=.1)
    plt.plot(t_eval, theta[:,N-1], dashes=dashes[1], lw=1,
            color="blue", alpha=1.)
    plt.plot(t_eval, theta[:,N], dashes=dashes[1], lw=1,
            color="red", alpha=1.)
    plt.ylabel(r'$\omega$')
    plt.xlabel(r'$t$')

    plt.plot([],[], dashes=dashes[0], lw=1, label=labels[0],
        color="blue", alpha=.1)
    plt.plot([],[], dashes=dashes[1], lw=1, label=labels[1],
        color="blue", alpha=1.)
    plt.plot([],[], dashes=dashes[1], lw=1, label=labels[2],
        color="red", alpha=1.)
    plt.plot([],[], dashes=dashes[0], lw=1, label=labels[3],
        color="red", alpha=.1)
    plt.ylabel(r'$\theta$')
    plt.xlabel(r'$t$')
    plt.legend(fontsize=legend_fontsize)

    fig.align_ylabels(axes)

    axi=3
    ax = fig.add_subplot(spec[axi//N_COL,axi%N_COL]) # row, col
    ax.text(.0, 1.03, fr"$\bf{{{string.ascii_uppercase[axi]}}}$", transform=ax.transAxes, size=12, weight="bold")
    axes.append(ax)

    # dashes = [(4., 4.), (2., .6)]
    for i in range(N-1):
        plt.plot(t_eval, omega[:,i], dashes=dashes[0], lw=1,
                color="blue", alpha=.1)
    for i in range(N+1,N*2):
        plt.plot(t_eval, omega[:,i], dashes=dashes[0], lw=1,
                color="red", alpha=.1)
    plt.plot(t_eval, omega[:,N-1], dashes=dashes[1], lw=1,
            color="blue", alpha=1.)
    plt.plot(t_eval, omega[:,N], dashes=dashes[1], lw=1,
            color="red", alpha=1.)
    plt.ylabel(r'$\omega$')
    plt.xlabel(r'$t$')

    plt.plot([],[], dashes=dashes[0], lw=1, label=labels[0],
        color="blue", alpha=.1)
    plt.plot([],[], dashes=dashes[1], lw=1, label=labels[1],
        color="blue", alpha=1.)
    plt.plot([],[], dashes=dashes[1], lw=1, label=labels[2],
        color="red", alpha=1.)
    plt.plot([],[], dashes=dashes[0], lw=1, label=labels[3],
        color="red", alpha=.1)
    # prop={'family':"Helvetica", 'size':4}
    plt.legend(fontsize=legend_fontsize)#, prop=prop)

    fig.align_ylabels(axes)
    # fname = f'Results/{"GG"}_{"hp"}-{PK[ENS,ENS2][0]:.1f}_{PK[ENS,ENS2][1]:04.1f}_{gamma}_{t_end}.png'
    # plt.savefig(fname=fname, format="png", bbox_inches="tight")