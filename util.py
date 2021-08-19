import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm
from scipy.optimize import basinhopping, brute, differential_evolution


# formerly compute_sym_graphmat_fast2
# increases speed over compute_sym_graphmat_slow by approx a factor of 10 on
# a 36x2 data matrix.
def compute_sym_graphmat(data, threshold):
    """Computes a graphmat object (Graph Matrix) denoting
    edge connections denoted by some threshold array.
    """
    n_cells, n_features = data.shape
    graphmat = np.zeros((n_cells, n_cells), dtype=int)

    bool_arr = np.zeros(data.shape, dtype=int)
    for i in range(n_cells):
        bool_arr[i, :] = data.iloc[i] > threshold

    for i in range(0, n_cells):
        for j in range(i, n_cells):
            if i != j:
                for k in range(n_features):
                    if nxor(bool_arr[i, k], bool_arr[j, k]):
                        graphmat[i, j] += 1

    graphmat += np.transpose(graphmat)

    return graphmat


def nxor(val_a, val_b):
    """Function to get not xor.

    nxor(True, True)   -> True

    nxor(True, False)  -> False

    nxor(False, True)  -> False

    nxor(False, False) -> True
    """
    if (val_a and val_b) or (not val_a and not val_b):
        return True
    else:
        return False


def plot_2dfeature_space(X, y, feature_names, class_names, thresholds=None):
    """ features is a Dict, all others are dataframes"""
    fig, ax = plt.subplots()
    colors = (np.unique(y, return_inverse=True)[1])
    scatter = ax.scatter(X[feature_names[0]], X[feature_names[1]], c=colors,
                         cmap = plt.get_cmap('jet'), alpha=0.3, s=215)
    ax.legend(handles=scatter.legend_elements()[0], labels=class_names)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])

    if thresholds is None:
        plt.title("Scatter plot of raw features")
    else:
        plt.title("Scatter plot of raw features with thresholds")

    if thresholds is not None:
        ax.axvline(thresholds[0], label="", color='k')
        ax.axhline(thresholds[1], label="", color='k')
    plt.show()


def find_feature_bounds(X):
    """Function for automatically gettings feature min/max bounds.

    Assumes that X is stored column-wise (each column represents
    different feature).
    """
    _, n_features = X.shape
    bounds = np.zeros((n_features, 2), dtype=float)

    for ti in range(n_features):
        # filter out things that are exactly zero
        # these are likely implemented in postprocessing
        # to mean that the feature is invalid
        non_zero_ind = np.where(X.iloc[:, ti] != 0)[0]
        bounds[ti, 0] = np.amin(X.iloc[non_zero_ind, ti])
        bounds[ti, 1] = np.amax(X.iloc[non_zero_ind, ti])

    # buffer so that data values are fully encapsulated in bound
    bounds[:, 0] -= 1e-8
    bounds[:, 1] += 1e-8

    return bounds


def get_average(arr):
    """ array is [ [x0 x1], [y0 y1], [z0 z1], ...]."""
    avg = [0.5*(arr[i][0] + arr[i][1]) for i in range(len(arr))]
    return avg


class MyBounds:
    def __init__(self, bounds):
        self.xmax = [bounds[i][1] for i in range(len(bounds))]
        self.xmin = [bounds[i][0] for i in range(len(bounds))]

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


def my_sim_anneal(X_train, bounds):
    def func(thresh):
        graphmat = compute_sym_graphmat(X_train, thresh)
        energy = compute_mod_energy(graphmat)
        # want to maximize 'energy', so need to return negative
        return -energy

    x0 = get_average(bounds)
    mybounds = MyBounds(bounds)
    minimizer_kwargs = {"method": "BFGS"}

    ret = basinhopping(func, x0, minimizer_kwargs=minimizer_kwargs, niter=20, accept_test=mybounds)

    return ret.x, ret.fun


def my_brute(X_train, bounds):
    def func(thresh):
        graphmat = compute_sym_graphmat(X_train, thresh)
        energy = compute_mod_energy(graphmat)
        # want to maximize 'energy', so need to return negative
        return -energy

    bounds2 = [(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
    x, fun, temp1, temp2 = brute(func, bounds2, Ns=12)

    return x, fun

def my_diff_evolution(X_train, bounds):
    def func(thresh):
        graphmat = compute_sym_graphmat(X_train, thresh)
        energy = compute_mod_energy(graphmat)
        # want to maximize 'energy', so need to return negative
        return -energy

    x0 = get_average(bounds)
    mybounds = MyBounds(bounds)
    minimizer_kwargs = {"method": "BFGS"}

    ret = differential_evolution(func, bounds, tol=1., maxiter=15, disp=True)

    return ret.x, ret.fun


def display_graphmat(graphmat):
    """Displays graphmat in a pop-up window."""
    plt.figure()
    plt.imshow(graphmat)
    plt.colorbar()
    plt.show()


def compute_total_edges(graphmat, max_features=2, power=1):
    """Computes total edges of a graphmat object."""
    return np.sum(np.power(graphmat, power))


def compute_mod_energy(graphmat):
    G = nx.Graph(graphmat)
    comm = next(nx_comm.girvan_newman(G))
    mod = nx_comm.modularity(G, comm)
    return mod


def get_color_map(G, comm):
    """gets color map for community `comm`."""
    COLORS = ['blue', 'green', 'red', 'yellow', 'purple', 'pink']
    n_comm = len(comm)
    color_map = []
    for node in G:
        for ind in range(n_comm):
            if node in comm[ind]:
                color_map.append(COLORS[ind])
                break
    
    return color_map


def get_pred(G, comm):
    """gets prediction for graph G and community partition `comm`."""
    n_comm = len(comm)
    pred = []
    for node in G:
        for group in range(n_comm):
            if node in comm[group]:
                pred.append(group)
                break
    
    return pred


def plot_2D_edges(edgesmat, feature_names):
    """Plots 2D edge matrix, with option to save.

    Color represents number of edges.

    Should give option to display threshold values along axis,
    as well as minimum.
    """
    N, _ = edgesmat.shape

    plt.figure()
    plt.imshow(edgesmat)
    plt.colorbar()
    plt.xlabel(f'{feature_names[0]} threshold index')
    plt.ylabel(f'{feature_names[1]} threshold index')
    plt.title("Total # of edges in matrix")

    ax = plt.gca()

    # Major ticks
    ax.set_xticks(np.arange(0, N, 3))
    ax.set_yticks(np.arange(0, N, 3))

    # Labels for major ticks
    ax.set_xticklabels(np.arange(1, N+1, 3))
    ax.set_yticklabels(np.arange(1, N+1, 3))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-.5, N, 1), minor=True)

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=0.4)

    plt.show()
