import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from scipy.stats import gamma, poisson, nbinom
from tqdm.notebook import tqdm

from scipy.optimize import minimize, curve_fit
from skopt import gp_minimize
from skopt.learning.gaussian_process import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from skopt.plots import plot_gaussian_process, expected_minimum

import sys
sys.path.insert(0, "../func_build/")
import like_func as lf
import data_utils as dtu


class gbm_infer_pars():
    """
    Parameters to pass to the likelihood MC computation
    """
    
    def __init__(self, dtimes, M_tot, n0, n1_min, n_eval, prior_pars, a_negbin=[], dt=0.1, n_discr=2000):
        
        # delta times between time steps
        self.dtimes = dtimes
        # Total number of cells
        self.M_tot = M_tot
        # Number of cells at birth
        self.n0 = n0
        # Minimum number of exp counts at first time to condition the inference
        self.n1_min = n1_min
        # Number of MC samples to evaluate the LL
        self.n_eval = n_eval
        # Priors on the gamma function parameters for the importance sampling
        self.prior_pars = prior_pars
        # a parameters of the neg-bin noise for each time step. If empty list we consider a Poisson
        self.a_negbin = a_negbin
        # Time step for the trajectory generation
        self.dt = dt
        # Number of bins of the log count space to compute the integral
        self.n_discr = n_discr
        
        
def stat_dist(x, gbm_pars):
    """
    Stationary distribution of the GBM
    """
    x0 = np.log(gbm_pars.n0)
    a = gbm_pars.alpha
    coef = 1 / x0
    return np.where(x < x0, coef * (1 - np.exp(-a*x)), coef * (np.exp(a*(x0-x)) - np.exp(-a*x)))


def birth_rate(M_n0_ratio, tau, theta):
    return M_n0_ratio * (1/tau - 1/theta/2)
    
    
def neg_bin_f(mu, a_negbin):
    """
    Negative binomial of the noise model with our parametrization and b = 1
    """
    mu[mu == 0] = 1e-5
    s2 = mu * (a_negbin + 1)
    return nbinom(mu * mu / (s2 - mu), mu / s2)


def get_gamma_prior(n_max, M1, M_tot, gamma_a, std_factor=3):
    """
    Gamma distribution prior of the importance sampling
    """
    x_max_eff = np.log(2*n_max*M_tot/M1)
    gamma_std = x_max_eff / std_factor
    return gamma(gamma_a, 0, scale=gamma_std/np.sqrt(gamma_a))


def compute_p_smaller_mmin(gbm_pars, M1, m_min, n_eval, noise_f):
    """
    Deterministic computation of the probability of observing less than m_min exp counts
    at the first time step
    """
    points = np.linspace(0, np.log(gbm_pars.M_tot), n_eval)
    dx = points[1] - points[0]
    mu1s = np.exp(points) * M1 / gbm_pars.M_tot
    aux = stat_dist(points, gbm_pars)
    ps = 0
    for m in range(m_min):
        ps += np.sum(aux * poisson(mu1s).pmf(m)) * dx
    return ps


def MC_gbm_full_ll(sp_counts, tau, theta, pars, import_f, build_probs=False):
    """
    LL computation using MC integration with importance sampling.
    """
    
    N_samp = int((len(sp_counts.columns) - 1) / 2.0)
    if len(pars.dtimes) != N_samp - 1:
        print("Invalid sample numbers")
        return
    
    if len(pars.a_negbin) == 0:
        a_nb = np.zeros(N_samp)
        noise_f = lambda mu, a, n : poisson(mu).pmf(n)
    else:
        a_nb = np.copy(pars.a_negbin)
        noise_f = lambda mu, a, n : neg_bin_f(mu, a).pmf(n)
    
    Ms = [np.sum(sp_counts["n"+str(i+1)] * sp_counts.occ) for i in range(N_samp)]

    xs_samp = [import_f.rvs(pars.n_eval)]
    for dtime in pars.dtimes:
        x_samp = np.array(lf.gen_gbm_traj(tau, theta, xs_samp[-1], dtime, pars.dt))
        xs_samp.append(x_samp)
        
    xs_bins, x_vals = dtu.discretize(xs_samp, pars.n_discr)
    for i in range(1, len(xs_bins)):
        xs_bins[i] = np.where(xs_samp[i] > 0, xs_bins[i] + 1, 0)

    gbm_pars = lf.gbm_likeMC_pars(tau, theta, pars.M_tot, pars.n0)
    aux = stat_dist(x_vals, gbm_pars) / import_f.pdf(x_vals)
    x_exp = np.exp(x_vals)
    mus = [Ms[0] * x_exp / gbm_pars.M_tot]
    for i in range(1, len(xs_bins)):
        mus.append(np.append(0, Ms[i] * x_exp / gbm_pars.M_tot))

    ll, tot_counts = 0, 0
    probs = pd.DataFrame(index=sp_counts.index, data={'prob' : np.zeros(len(sp_counts), dtype=int)})
    for _id, row in sp_counts.iterrows():

        if row.n1 < pars.n1_min: 
            continue

        ps = [(aux * noise_f(mus[0], a_nb[0], row.n1))[xs_bins[0]]]
        for i in range(1, len(xs_bins)):
            ps.append(noise_f(mus[i], a_nb[i], row['n'+str(i+1)])[xs_bins[i]])
            
        p = np.sum(np.prod(ps, axis=0)) / len(xs_samp[0])
        if p > 0:
            ll += np.log(p) * row.occ
            tot_counts += row.occ
            probs.loc[_id] = p
    
    noise0_f = lambda mu, n : noise_f(mu, a_nb[0], n)
    p_ext = 1 - compute_p_smaller_mmin(gbm_pars, Ms[0], pars.n1_min, pars.n_eval * 5, noise0_f)
    probs.prob /= p_ext
    ll = (ll - tot_counts * np.log(max(1e-300, p_ext))) / tot_counts 
    
    if build_probs:
        return ll, probs
    else:
        return ll


def compute_sp_count_probs(sp_counts, tau, theta, pars, max_n):
    """
    It fills the sp_counts with all the combinations of counts smaller than max_n.
    """
    d = int((len(sp_counts.columns) - 1) / 2.0)
    prod_set = [np.arange(max_n)]*d
    for ns in product(*prod_set):
        _id = ""
        col_dic = dict()
        for i, n in enumerate(ns):
            col_dic['n'+str(i+1)] = n
            _id += str(n) + "_"
        _id = _id[:-1]

        if _id not in sp_counts.index:
            sp_counts.loc[_id] = col_dic
        
    max_n1 = max(sp_counts.n1.values)
    M1 = np.sum(sp_counts.n1 * sp_counts.occ)
    import_f = get_gamma_prior(max_n1, M1, pars.M_tot, pars.prior_pars[0], pars.prior_pars[1])
    ll, probs = MC_gbm_full_ll(sp_counts, tau, theta, pars, import_f, build_probs=True)
    for i in range(d):
        probs['n' + str(i+1)] = np.array(probs.index.str.split('_').str[i], dtype=int)
    return probs[probs['n1'] >= pars.n1_min]


def scan_gamma_prior_std(gamma_as, std_factors, max_n1, M1, M_tot, like_f_at_prior, R):
    """
    Evaluate the LL at different parameters of the gamma prior for the importance sampling
    """
    progress = tqdm(total=len(gamma_as)*len(std_factors))
    stds = np.zeros((len(gamma_as), len(std_factors)))
    avs = np.zeros((len(gamma_as), len(std_factors)))
    for i, a in enumerate(gamma_as):
        for j, f in enumerate(std_factors):
            import_f = get_gamma_prior(max_n1, M1, M_tot, a, f)
            vals = [like_f_at_prior(import_f) for _ in range(R)]
            stds[i,j] = np.std(vals) / np.mean(vals)
            avs[i,j] = np.mean(vals)
            progress.update(1)
            
    return avs, stds
        
        
def ll_to_min(x, sp_counts, inf_p):
    """
    Function to minimize in the optimization methods
    """
    tau, theta = x[0], x[1]
    aux_pars = lf.gbm_likeMC_pars(tau, theta, inf_p.M_tot, inf_p.n0)
    M1 = np.sum(sp_counts.n1 * sp_counts.occ)
    max_n1 = max(sp_counts.n1.values)
    import_f = get_gamma_prior(max_n1, M1, inf_p.M_tot, inf_p.prior_pars[0], inf_p.prior_pars[1])
    return -MC_gbm_full_ll(sp_counts, tau, theta, inf_p, import_f)


def learn_gbm_gp_alpha(sp_counts, tau, infer_pars, bounds=[(1., 2.0)], n_calls=200, noise=1e-2, 
                 lenght_scales=1.0, xi=1e-3, n_jobs=1):
    """
    Gaussian process optimization for alpha at given tau.
    """
    ll_func = lambda x : ll_to_min((tau, x[0]*tau/2.0), sp_counts, infer_pars)
    opt_result = learn_gp(ll_func, bounds, n_calls, noise, [lenght_scales], xi, n_jobs)
    return opt_result


def learn_gbm_gp_tau(sp_counts, alpha, infer_pars, bounds=[(7.0, 30.0)], n_calls=200, noise=1e-2, 
                 lenght_scales=50.0, xi=3e-4, n_jobs=1, log_tau=False):
    """
    Gaussian process optimization for alpha at given tau.
    """
    if log_tau:
        ll_func = lambda x : ll_to_min((np.exp(x[0]), alpha*np.exp(x[0])/2.0), sp_counts, infer_pars)
        bounds = [(np.log(bounds[0][0]), np.log(bounds[0][1]))]
    else:
        ll_func = lambda x : ll_to_min((x[0], alpha*x[0]/2.0), sp_counts, infer_pars)
        
    opt_result = learn_gp(ll_func, bounds, n_calls, noise, [lenght_scales], xi, n_jobs)
    if log_tau:
        opt_result.x[0] = np.exp(opt_result.x[0])
    
    return opt_result


def learn_gbm_gp(sp_counts, infer_pars, bounds=[(7.0, 30.0), (1., 2.0)], n_calls=200, noise=1e-2, 
                 lenght_scales=[100.0, 1.0], xi=2e-4, n_jobs=1, log_tau=False):
    """
    Gaussian process optimization for tau and alpha.
    """
    if log_tau:
        ll_func = lambda x : ll_to_min((np.exp(x[0]), x[1]*np.exp(x[0])/2.0), sp_counts, infer_pars)
        bounds = [(np.log(bounds[0][0]), np.log(bounds[0][1])), bounds[1]]
    else:
        ll_func = lambda x : ll_to_min((x[0], x[1]*x[0]/2.0), sp_counts, infer_pars)
    
    opt_result = learn_gp(ll_func, bounds, n_calls, noise, lenght_scales, xi, n_jobs)
    if log_tau:
        opt_result.x[0] = np.exp(opt_result.x[0])
        
    return opt_result


def learn_gp(ll_func, bounds, n_calls, noise, lenght_scales, xi, n_jobs, cb=lambda res : None):
    
    l_bounds = (min(lenght_scales)*1e-2, max(lenght_scales)*1e2) 
    # nu=2.5 for twice differentiable function
    matern = Matern(length_scale=lenght_scales, length_scale_bounds=l_bounds, nu=2.5)
    n_bounds = (noise*1e-2, noise*1e2)
    w_kern = WhiteKernel(noise_level=noise, noise_level_bounds=n_bounds)
    kernel = ConstantKernel(1.0, (0.1, 10.0)) * matern + w_kern
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5, noise='gaussian')

    res = gp_minimize(ll_func, base_estimator=gp, dimensions=bounds, n_calls=n_calls, \
                      acq_func='EI', xi=xi, n_jobs=n_jobs, callback=cb)
    best_p, fun = expected_minimum(res)
    res.fun = fun
    res.x = best_p
    return res


def plot_heatmap(ax, mat, xs, ys, vmin, vmax, labx='theta', laby='tau', cb=True):
    ax.set_xlabel(labx, fontsize=12)
    ax.set_ylabel(laby, fontsize=12)
    ax.set_xticks(np.arange(0, len(xs)))
    ax.set_xticklabels(["%2.1f"%s for s in xs])
    ax.set_yticks(np.arange(0, len(ys)))
    ax.set_yticklabels(["%2.1f"%s for s in ys[::-1]])
    im = ax.imshow(mat[::-1], vmin=vmin, vmax=vmax)
    if cb:
        cb = plt.colorbar(im, ax=ax)
    return ax, cb