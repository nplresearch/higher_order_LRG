import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import t 
import scipy
from Functions import scomplex
from Functions import plotting
import powerlaw as pwl
import pickle

plt.rcParams['text.usetex'] = True
palette = np.array([[0.3647,    0.2824,   0.1059],
           [0.8549,    0.6314,    0.3294],
           [0.4745,    0.5843,    0.5373],
           [0.4745,    0.3843,    0.7373]])


N = 100
n_tau = 30#100
rep = 2#10#5
d = 1
s = 1
METHOD = "representative"
SPARSIFY = False
TRUE_CONNECTIONS = True
beta = 0.1
ls = True # logscale

pref = "d" + str(d) + "s" + str(s)
path = "/Tests/Experiments_"+ str(METHOD)+ "_"+ str(SPARSIFY)+ "_"+ str(TRUE_CONNECTIONS)+"/"+pref


def plot_mean(Ns, data, col, logscale,ax,lab):
    Nmin = np.min(np.min(Ns))
    Nmax = np.max(np.max(Ns))
    xq = np.linspace(Nmin, Nmax, 100)
    dataq = np.zeros((Ns.shape[0], 100))
    for r in range(Ns.shape[0]):
        uNs, uid = np.unique(Ns[r, :], return_index=True)
        dataq[r, :] = np.interp(xq, uNs, data[r, uid], left=0, right=0)
    if logscale:
        ax.semilogy(xq, np.mean(dataq, axis=0), linewidth=2, color=col, label = lab)
    else:
        ax.plot(xq, np.mean(dataq, axis=0), linewidth=2, color=col, label = lab)


#with open(path+'/Ns.pkl', 'rb') as f:
#    Ns = pickle.load(f)
with open(path+'/deg_dist.pkl', 'rb') as f:
    deg_dist = pickle.load(f)


Ns = np.zeros((rep,d+1,n_tau),dtype=int)
for r in range(rep):
    for i in range(d+1):
        for t in range(n_tau):
            Ns[r,i,t] = len(deg_dist[r][t][i][0])

#Ns = sp.io.loadmat(path + '/Ns.mat')['Ns']
#deg_dist = sp.io.loadmat(fpath pref + '/deg_dist.mat')['deg_dist']

names = ["Node", "Link", "Face", "Tetrahedron"]


deg_distance = np.zeros((rep,d+1,d,n_tau))
for r in range(rep):
    for norml in range(d + 1):
        for degg in range(d):
            deg1 = deg_dist[r][0][0][degg]
            for tau in range(n_tau):
                deg2 = deg_dist[r][tau][norml][degg]
                if len(deg2) == 0:
                    deg2 = [0]
                # KS Distance
                test = scipy.stats.kstest(deg1,deg2)
                deg_distance[r, norml, degg, tau] = test.statistic
                # Powerlaw
                #fit_function = pwl.Fit(deg2)
                #deg_distance[r, norml, degg, tau] = fit_function.power_law.D



fig = plt.figure()

gs = fig.add_gridspec(d,2)
ax1 = fig.add_subplot(gs[:, 0])
axv = []
for i in range(d):
    ax = fig.add_subplot(gs[i, 1])
    axv = axv+[ax]
fig.tight_layout()

if d == 1:
    sc = scomplex.NGF_d1(100, s, beta)
elif d == 2:
    sc = scomplex.NGF_d2(100, s, beta)
elif d == 3:
    sc = scomplex.NGF_d3(100, s, beta)

plotting.plot_complex(sc,ax1)
ax1.set_title(r"\textbf{NGF d = " + str(d) + ", s = " + str(s) + "}", fontsize=14)
for i in range(d):
    ax = axv[i]
    for j in range(d + 1):
        lab = "L" + str(j)
        plot_mean(1 - Ns[:, j, :] / N, deg_distance[:, j, i, :], palette[j, :], ls, ax,lab)
        if ls:
            for r in range(rep):
                ax.semilogy(1 - Ns[r, j, :] / N, deg_distance[r, j, i, :], linewidth=1, color= palette[j, :], alpha = 0.3)
        else:
            for r in range(rep):
                ax.plot(1 - Ns[r, j, :] / N, deg_distance[r, j, i, :], linewidth=1, color=palette[j, :], alpha = 0.3)

    ax.legend()
    ax.set_title(r"\textbf{" + names[i] + "-" + names[d] + "}", fontsize=14)
    ax.set_xlim([0, 0.8])

plt.show()