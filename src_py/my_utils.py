import numpy as np
from multiprocessing import Pool, cpu_count

    
def generate_plaw_dist_log(alpha, f_min, n_freqs):
    """
    It generates n_freq log-spaced frequencies from f_min to 1
    that follow a power law distribution with exponent -alpha.
    It returns the frequencies and the probabilities of extracting them
    """
    freqs_bin = np.logspace(np.log10(f_min), 0, n_freqs+1)
    dfs_log = freqs_bin[1:] - freqs_bin[:-1]
    freqs = freqs_bin[1:] - dfs_log/2
    weights = freqs**(-alpha) * dfs_log
    norm = weights.sum()
    probs = weights / norm
    return freqs, probs


def parallelize_async(func, args_list, n_threads, chunksize=1):
    """
    Parallelize the given function in n_threads. Each process correspond to an item in args_list
    which is passed to the function.
    Parallelization using pool.map_async: the returned list of result is not necessarily ordered as
    args_list.
    Important: args have to be an iterable. In the case of single argument use "(arg,)".
    """
    n_threads = min(n_threads, cpu_count())
    pool = Pool(n_threads)
    results = pool.map_async(func, args_list, chunksize=chunksize)
    pool.close()
    pool.join()
    return [r for r in results.get()]


def log_like(ns_uniq, ns_count, pars, integrator):
    """
    Computing the log likelihood from the sparse count representation
    """
    
    like_0 = integrator(pars, np.zeros(len(ns_uniq[0]), dtype=int))
    N_obs = sum(ns_count)
    N = N_obs / (1 - like_0)
    
    ll = 0
    for ns, n_count in zip(ns_uniq, ns_count):
        like_n = integrator(pars, ns)
        ll += np.log(like_n) * n_count
        
    return ll - N_obs * np.log(max(1 - like_0, 1e-30)), N


def hess(f, x, dx=[0.01, 0.01]):
    """
    Numerical evaluation of the Hessian matrix of the function f
    """
    hess = np.zeros((len(x), len(x)))
    for d1 in range(len(x)):
        for d2 in range(len(x)):
            x1, x2, x3, x4 = [xv for xv in x], [xv for xv in x], [xv for xv in x], [xv for xv in x]
            x1[d1] = x1[d1] + dx[0]/2.0
            x1[d2] = x1[d2] + dx[1]/2.0
            x2[d1] = x2[d1] + dx[0]/2.0
            x2[d2] = x2[d2] - dx[1]/2.0
            x3[d1] = x3[d1] - dx[0]/2.0
            x3[d2] = x3[d2] + dx[1]/2.0
            x4[d1] = x4[d1] - dx[0]/2.0
            x4[d2] = x4[d2] - dx[1]/2.0
            hess[d1, d2] = (f(x1) - f(x2) - f(x3) + f(x4) ) / dx[0] / dx[1]
    return hess