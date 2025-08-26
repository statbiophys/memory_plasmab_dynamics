import numpy as np
from multiprocessing import Pool, cpu_count
import sys
sys.path.insert(0, "../func_build/")
import like_func as lf
import data_utils as dt

    

#def generate_plaw_dist_log(alpha, f_min, n_freqs):
#    """
#    It generates n_freq log-spaced frequencies from f_min to 1
#    that follow a power law distribution with exponent -alpha.
#    It returns the frequencies and the probabilities of extracting them
#    """
#    freqs_bin = np.logspace(np.log10(f_min), 0, n_freqs+1)
#    dfs_log = freqs_bin[1:] - freqs_bin[:-1]
#    freqs = freqs_bin[1:] - dfs_log/2
#    weights = freqs**(-alpha) * dfs_log
#    norm = weights.sum()
#    probs = weights / norm
#    return freqs, probs


# def generate_samples(noise_gen, N_cells, n_threads=8):
#     noise_gen.sample_repertoire()
#    samples, M_eff = [], []
#    for M in N_cells:
#        samples.append(np.array(noise_gen.sample(M, n_threads)))
#        M_eff.append(np.sum(samples[-1]))
#    sparse_counts = dt.build_sparse_counts(samples)
#    n_uniq = sparse_counts[['n'+str(i+1) for i in range(len(N_cells))]].values
#    return n_uniq, sparse_counts['occ'].values, np.array(M_eff)

    
# Used by 7 Phad, Mikelov
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


def grad(f, x, dx=0.001):
    """
    Numerical evaluation of the gradient of a function f
    """
    if type(dx) == float:
        dx = np.ones(len(x)) * dx
        
    grad = np.zeros(len(x))
    for d in range(len(x)):
        x1, x2 = [xv for xv in x], [xv for xv in x]
        x1[d] = x1[d] - dx[d]/2.0
        x2[d] = x2[d] + dx[d]/2.0
        grad[d] = (f(x2) - f(x1)) / dx[d]
            
    return grad


# Used by 7 Phad, Mikelov
def compute_errors(H, constr_func, opt_x, dx=0.001):
    """
    Error estimates obtained by projecting the Hessian of the loglikelihood
    on the contraint, diagonalizing the projection and taking the inverse
    of the eigenvalues. The error for each parameter is the square root of
    the sum of the eigenvalue inverse weighted by the corresponding eigenvector
    component along the parameter axis.
    """
    
    # Finding the gradient of the constraint
    grad_c = grad(constr_func, opt_x, dx)
    # Normal to the contsraint
    n = grad_c / np.linalg.norm(grad_c)
    
    # Projection matrix to the normal to the constraint
    P_n = np.array(n, ndmin=2).T @ np.array(n, ndmin=2)
    # Projection matrix to the contraint
    P_constr = np.identity(len(P_n)) - P_n
    # Hessian projected onto the constraint
    H_constr = P_constr @ H @ P_constr
    # Diagonalizing
    ei_val, ei_vec = np.linalg.eig(H_constr)
    ei_vec_inv = np.linalg.inv(ei_vec)
    
    # Considering only the non-zero eivals (one zero eival of the eigenvec normal to the grad)
    nozero_ei_ind = np.argsort(np.abs(ei_val))[1:]
    errs = np.zeros(len(ei_vec))
    for i in nozero_ei_ind:
        errs += ei_vec[:,i]*ei_vec_inv[i,:] / ei_val[i]
    # The errors (before sqrt) are the diagonal elements of the inverse matrix written as 
    # an eigenvalue decomposition but without the eigenvalue corresponding to the direction
    # perpendicular to the constraint
    return np.sqrt(errs)


def hess(f, x, dx=0.001):
    """
    Numerical evaluation of the Hessian matrix of the function f. Very unstable for loglike!
    """
    if type(dx) == float:
        dx = np.ones(len(x)) * dx
        
    hess = np.zeros((len(x), len(x)))
    for d1 in range(len(x)):
        for d2 in range(d1+1):
            x1, x2 = np.array([xv for xv in x]), np.array([xv for xv in x])
            x3, x4 = np.array([xv for xv in x]), np.array([xv for xv in x])
            x1[d1] = x1[d1] + dx[d1]/2.0
            x1[d2] = x1[d2] + dx[d2]/2.0
            x2[d1] = x2[d1] + dx[d1]/2.0
            x2[d2] = x2[d2] - dx[d2]/2.0
            x3[d1] = x3[d1] - dx[d1]/2.0
            x3[d2] = x3[d2] + dx[d2]/2.0
            x4[d1] = x4[d1] - dx[d1]/2.0
            x4[d2] = x4[d2] - dx[d2]/2.0
            hess[d1, d2] = ( f(x1) - f(x2) - f(x3) + f(x4) ) / dx[d1] / dx[d2]
            hess[d2, d1] = hess[d1, d2]
    return hess


def get_pn_at_m_minus(samp_index, m_minus, R, sp_count_fr):
    """
    Given a sparse count frame, it returns the probability list of observing counts 
    in the sample of given index given the summed count (m_minus) of all the other
    samples
    """
    n_other_labels = ['n'+str(i+1) for i in range(R) if i+1 != samp_index]
    sp_count_fr['m_minus_i'] = sp_count_fr[n_other_labels].sum(axis=1)
    aux = sp_count_fr[sp_count_fr['m_minus_i'] == m_minus]
    aux = aux.groupby('n'+str(samp_index)).agg(sum)
    return aux.index.values, aux['occ'].values / np.sum(aux['occ'])


def gp_mean(x, gp_res):
    x = gp_res.space.transform(np.array(x).reshape(1, -1))
    return gp_res.models[-1].predict(x.reshape(1, -1))[0]


def error_prod(prod, a, b, erra, errb, covab):
    return prod*np.sqrt((erra/a)**2 + (errb/b)**2 + 2*covab/a/b)



#def log_like(ns_uniq, ns_count, pars, integrator):
#    """
#    Computing the log likelihood from the sparse count representation
#    """
#    
#    like_0 = integrator(pars, np.zeros(len(ns_uniq[0]), dtype=int))
#    N_obs = sum(ns_count)
#    N = N_obs / (1 - like_0)
#    
#    ll = 0
#    for ns, n_count in zip(ns_uniq, ns_count):
#        like_n = integrator(pars, ns)
#        ll += np.log(max(like_n, 1e-300)) * n_count
#        
#    return ll - N_obs * np.log(max(1 - like_0, 1e-300)), N


#def hess_plawpoiss_M(ns_uniq, ns_count, beta, fmin, N_cells):
#    """
#    Hessian of the log likelihood of the power-law poisson integral with transformed variables:
#    log10 fmin, log10 M. Much faster and reliable than the numerical evaluation with hess.    
#    """
#    hess = np.zeros((2+len(N_cells),2+len(N_cells)))
#    pars = lf.plaw_poiss_pars(beta, fmin, N_cells)
#    for ns, n_count in zip(ns_uniq, ns_count):
#        hess += np.array(lf.hess_log_integr_plaw_poiss_M(pars, ns)) * n_count
#    hess -= np.array(lf.hess_log_integr_plaw_poiss_M(pars, np.zeros(len(N_cells), dtype=int))) * np.sum(ns_count)
#    return hess

#def hess_plawpoiss(ns_uniq, ns_count, beta, fmin, N_cells):
#    """
#    Hessian of the log likelihood of the power-law poisson integral with transformed variables:
#    log10 fmin. Much faster and reliable than the numerical evaluation with hess.    
#    """
#    hess = np.zeros((3,3))
#    pars = lf.plaw_poiss_pars(beta, fmin, N_cells)
#    for ns, n_count in zip(ns_uniq, ns_count):
#        hess += np.array(lf.hess_log_integr_plaw_poiss(pars, ns)) * n_count
#    hess -= np.array(lf.hess_log_integr_plaw_poiss(pars, np.zeros(len(N_cells), dtype=int))) * np.sum(ns_count)
#    return hess


#def jac_plawpoiss(ns_uniq, ns_count, beta, fmin, N_cells):
#    """
#    Jacobian of the log likelihood of the power-law poisson integral with transformed variables:
#    log10 fmin.
#    """
#    g = np.zeros(3)
#    pars = lf.plaw_poiss_pars(beta, fmin, N_cells)
#    for ns, n_count in zip(ns_uniq, ns_count):
#        g += np.array(lf.jac_log_plaw_poiss(pars, ns)) * n_count
#    z = np.zeros(len(N_cells), dtype=int)
#    g -= np.array(lf.jac_log_plaw_poiss(pars, z)) * np.sum(ns_count)
#    return g




#def hess_plawnegbin(ns_uniq, ns_count, nb_pars):
#    """
#    Hessian of the log likelihood of the power-law negbin integral with transformed variables:
#    log10 fmin. Much faster and reliable than the numerical evaluation with hess.    
#    """
#    hess = np.zeros((5, 5))
#    for ns, n_count in zip(ns_uniq, ns_count):
#        hess += np.array(lf.hess_log_integr_plaw_negbin(nb_pars, ns)) * n_count
#    hess -= np.array(lf.hess_log_integr_plaw_negbin(nb_pars, np.zeros(len(nb_pars.Ms), dtype=int))) * np.sum(ns_count)
#    return hess


#def jac_plawnegbin(ns_uniq, ns_count, beta, fmin, a, b, N_cells):
#    """
#    Jacobian of the log likelihood of the power-law poisson integral with transformed variables:
#    log10 fmin.
#    """
#    g = np.zeros(5)
#    pars = lf.plaw_negbin_pars(beta, fmin, a, b, N_cells)
#    for ns, n_count in zip(ns_uniq, ns_count):
#        g += np.array(lf.jac_log_plaw_negbin(pars, ns)) * n_count
#    z = np.zeros(len(N_cells), dtype=int)
#    g -= np.array(lf.jac_log_plaw_negbin(pars, z)) * np.sum(ns_count)
#    return g


