import numpy as np
import pandas as pd
import my_utils as ut
from scipy.optimize import minimize
from pathlib import Path
import sys
sys.path.insert(0, "../func_build/")
import like_func as lf



def compute_pn(counts, beta, fmin, a, b, Ms):
    """
    Compute the probability of observing the list of counts ns from a neg-bin power law
    noise model.
    """
    P_ns = np.zeros(len(counts))
    pars = lf.plaw_negbin_pars(beta, fmin, a, b, [sum(Ms)])
    for i, ns in enumerate(counts):
        P_ns[i] = lf.integr_plaw_negbin(pars, [ns])
    P0 = lf.integr_plaw_negbin(pars, [0])
    return P_ns / (1-P0)


def compute_pn_at_m_minus(counts, m_minus, R, beta, fmin, a, b, M_fix):
    """
    It computes the probabily of observing the counts given the summation m_minus in all the 
    other samples in a power-law negbin model.
    All the samples need to have the same size M_fix.
    """
    P_ns, P_ni_and_ns = 0, np.zeros(len(counts))
    Ms = np.ones(R) * M_fix
    for ns_set in _generate_N_tuple_sum_S(R-1, m_minus):
        m = _tuple_multiplicity(ns_set)
        pars = lf.plaw_negbin_pars(beta, fmin, a, b, Ms[:-1])
        P_ns += m * lf.integr_plaw_negbin(pars, list(ns_set))
        for i, ni in enumerate(counts):
            pars = lf.plaw_negbin_pars(beta, fmin, a, b, Ms)
            P_ni_and_ns[i] += m * lf.integr_plaw_negbin(pars, np.append(ni, list(ns_set)))
        
    if m_minus == 0:
        pars = lf.plaw_negbin_pars(beta, fmin, a, b, [Ms[0]])
        P0 = lf.integr_plaw_negbin(pars, [0])
        P_ni_and_ns /= (1-P0)
        
    return P_ni_and_ns / P_ns



class infer_noise() :
    """
    Abstract class performing the noise inference of different replicates of a 
    power-law distributed system.
    It requires a sparse representation of count-tuples and their multiplicity.
    """
    
    def __init__(self, n_uniq, n_counts, n_points=5000, verbose=False):
        self.n_uniq = n_uniq
        self.n_counts = n_counts
        self.R = len(self.n_uniq[0])
        self.N_cell_obs = np.array([np.sum(ns * n_counts) for ns in n_uniq.T])
        self.N_clones = np.sum(n_counts)
        self.n_points = n_points
        self.verbose = verbose
        self.name = ''
        self.result, self.errors = None, None
        self.eps_SLSQP=10**(-2)
        self.ftol=10**(-5)
        
        
    def run(self, x0=(), bounds=()):
        
        if self.name == '':
            print('You are running the abstract class')
            return
        
        self.N_storage = dict()
        self.cb = lambda x : print(x)
        if not self.verbose: self.cb = lambda x : None
            
        if len(x0) == 0: x0 = self.default_x0
        if len(bounds) == 0: bounds = self.default_bounds
            
        self.result = minimize(self._ll_func, x0=x0, bounds=bounds, 
                  constraints={'type':'eq', 'fun':self._constr_func}, 
                  method='SLSQP', options={'eps':self.eps_SLSQP, 'ftol':self.ftol, 'maxiter':200},
                  callback = self.cb
        )
        return self.result
        
        
    def compute_errors(self):
        if not self._check_run(): return
        self.H = self._compute_hessian()
        self.errors = ut.compute_errors(-self.H, self._constr_func, self.result.x)
        return self.errors


    def write_on_file(self, folder, name=''):
        if name == '': name='infer_noise_'+self.name+'.txt'
        if not self._check_run(): return
        Path(folder).mkdir(parents=True, exist_ok=True)
        f = open(folder+name, 'w')
        f.write('success:\t' + str(self.result.success) + str('\n'))
        f.write('minus_ll:\t' + str(self.result.fun) + str('\n'))
        f.write('n_points:\t' + str(self.n_points) + str('\n'))
        f.write('eps:\t' + str(self.eps_SLSQP) + str('\n'))
        f.write('ftol:\t' + str(self.ftol) + str('\n'))
        f.write('best parameters:\n')
        for p_name, val in zip(self.pars_names, self.result.x):
            f.write(p_name + ':\t' + str(val) + str('\n'))
        if self.errors is not None:
            f.write('errors:\n')
            for p_name, val in zip(self.pars_names, self.errors):
                f.write(p_name + ':\t' + str(val) + str('\n'))
        f.close()
        
        
    def _check_run(self):
        if self.result is None:
            print('Run the inference first')
            return False
        return True
        
        
    def _get_constr(self, pars_tuple, pars, integrator):
        if pars_tuple in self.N_storage:
            N = self.N_storage[pars_tuple]
        else:
            like_0 = integrator(pars, np.zeros(len(self.n_uniq[0]), dtype=int), n_points=self.n_points)
            N = self.N_clones / (1 - like_0)
        return lf.plaw_average(pars.beta, pars.fmin)*N - 1
    
    
class infer_noise_poiss(infer_noise):
    """
    Implementation of the abstract class for the poisson model of noise. infer_norm=True
    is a modified version that multiply all the experimental sizes by a factor (equivalent
    to imagine that the frequencies are not normalized on the realization)
    """
    def __init__(self, n_uniq, n_counts, infer_norm, n_points=10000, verbose=False):
        super().__init__(n_uniq, n_counts, n_points, verbose)
        self.infer_norm = infer_norm
        self.name = 'poisson'
        self.pars_names = ['beta', 'fmin']
        self.default_x0 = (2.3, -5.5)
        self.default_bounds = ((1.8, 3), (-9, -4))
        if infer_norm: 
            self.name += '_norm'
            self.pars_names.append('c')
            self.default_x0 += (1,)
            self.default_bounds += ((0.5, 2),)
            
            
    def _ll_func(self, pars_to_min):
        
        beta, fmin, c = pars_to_min[0], 10**pars_to_min[1], 1
        if self.infer_norm: c = pars_to_min[2]
            
        pars = lf.plaw_poiss_pars(beta, fmin, self.N_cell_obs * c)
        like_0 = lf.integr_plaw_poiss(pars, np.zeros(self.R, dtype=int), self.n_points) 
        self.N_storage[(beta, fmin, c)] = self.N_clones / (1 - like_0)
        ll = 0
        for ns, n_count in zip(self.n_uniq, self.n_counts):
            like_n = lf.integr_plaw_poiss(pars, ns, self.n_points)
            ll += np.log(max(like_n, 1e-300)) * n_count
        ll -= self.N_clones * np.log(max(1 - like_0, 1e-300))
        
        return -ll / self.N_clones

    
    def _constr_func(self, pars_to_min):
        beta, fmin, c = pars_to_min[0], 10**pars_to_min[1], 1
        if self.infer_norm: c = pars_to_min[2]
        pars = lf.plaw_poiss_pars(beta, fmin, self.N_cell_obs * c)
        return self._get_constr((beta, fmin, c), pars, lf.integr_plaw_poiss)

    
    def _compute_hessian(self):
        c = 1
        if self.infer_norm: c = self.result.x[2]
        
        self.hess = np.zeros((3,3))
        pars = lf.plaw_poiss_pars(self.result.x[0], 10**self.result.x[1], self.N_cell_obs*c)
        for ns, n_count in zip(self.n_uniq, self.n_counts):
            self.hess += np.array(lf.hess_log_integr_plaw_poiss(pars, ns, n_points=self.n_points)) * n_count
        zeros = np.zeros(self.R, dtype=int)
        self.hess -= np.array(lf.hess_log_integr_plaw_poiss(pars, zeros, n_points=self.n_points)) * self.N_clones

        if self.infer_norm: return self.hess
        else: return self.hess[[0,1]][:,[0,1]]


class infer_noise_negbin(infer_noise):
    """
    Implementation of the abstract class for the negative binomial model of noise. 
    infer_b sets if the second parameter is learned or chose equal to 1.
    infer_norm=True is a modified version that multiply all the experimental sizes by 
    a factor (equivalent to imagine that the frequencies are not normalized on the realization)
    """
        
    def __init__(self, n_uniq, n_counts, infer_b, infer_norm, n_points=10000, verbose=False, n_threads=8):
        super().__init__(n_uniq, n_counts, n_points, verbose)
        self.infer_norm = infer_norm
        self.infer_b = infer_b
        self.n_threads = n_threads
        self.name = 'negbin'
        self.pars_names = ['beta', 'fmin', 'a']
        self.default_x0 = (2.3, -5.5, 0.1)
        self.default_bounds = ((2, 3), (-9, -4), (0.001, 1))
        if infer_b:
            self.name += '_b'
            self.pars_names.append('b')
            self.default_x0 += (1,)
            self.default_bounds += ((0.5, 2),)
        if infer_norm: 
            self.name += '_norm'
            self.pars_names.append('c')
            self.default_x0 += (1,)
            self.default_bounds += ((0.5, 2),)
            
            
    def _ll_func(self, pars_to_min):
        
        beta, fmin, a, b, c = pars_to_min[0], 10**pars_to_min[1], pars_to_min[2], 1, 1
        if self.infer_b: b = pars_to_min[3]
        if self.infer_norm: c = pars_to_min[-1]
            
        pars = lf.plaw_negbin_pars(beta, fmin, a, b, self.N_cell_obs * c)
        zs = np.zeros(self.R, dtype=int)
        like_0 = lf.integr_plaw_negbin(pars, zs, self.n_points)
        self.N_storage[(beta, fmin, a, b, c)] = self.N_clones / (1 - like_0)
        
        args = [(ns, n_count, beta, fmin, a, b, self.N_cell_obs * c, self.n_points) 
                for ns, n_count in zip(self.n_uniq, self.n_counts)]
        lls = ut.parallelize_async(_aux_ll_negbin_parall, args, self.n_threads)
        ll = np.sum(lls) - self.N_clones * np.log(max(1 - like_0, 1e-30))
        return -ll / self.N_clones


    def _constr_func(self, pars_to_min):
        beta, fmin, a, b, c = pars_to_min[0], 10**pars_to_min[1], pars_to_min[2], 1, 1
        if self.infer_b: b = pars_to_min[3]
        if self.infer_norm: c = pars_to_min[-1]
        pars = lf.plaw_negbin_pars(beta, fmin, a, b, self.N_cell_obs * c)
        return self._get_constr((beta, fmin, a, b, c), pars, lf.integr_plaw_negbin)
    
    
    def _compute_hessian(self):
        beta, fmin, a, b, c = self.result.x[0], 10**self.result.x[1], self.result.x[2], 1, 1
        ind = [0,1,2]
        if self.infer_b: 
            b = self.result.x[3]
            ind.append(3)
        if self.infer_norm: 
            c = self.result.x[-1]
            ind.append(4)
            
        args = [(ns, n_count, beta, fmin, a, b, c*self.N_cell_obs) for ns, n_count in zip(self.n_uniq, self.n_counts)]
        hess = np.sum(ut.parallelize_async(_aux_hess_paral_negbin, args, self.n_threads), axis=0)
        pars = lf.plaw_negbin_pars(beta, fmin, a, b, c*self.N_cell_obs)
        hess -= np.array(lf.hess_log_integr_plaw_negbin(pars, np.zeros(self.R, dtype=int))) * self.N_clones
        return hess[ind][:,ind]
    
    
def _aux_ll_negbin_parall(args):
    ns, n_count, beta, fmin, a, b, N_reads, n_points = args
    pars = lf.plaw_negbin_pars(beta, fmin, a, b, N_reads)
    like_n = lf.integr_plaw_negbin(pars, ns, n_points)
    return np.log(max(like_n, 1e-300)) * n_count


def _aux_hess_paral_negbin(args):
    ns, n_count, beta, fmin, a, b, N_reads = args
    pars = lf.plaw_negbin_pars(beta, fmin, a, b, N_reads)
    return np.array(lf.hess_log_integr_plaw_negbin(pars, ns)) * n_count


def _sum_to_n(n, size, limit=None):
    """
    Produce all lists of `size` positive integers in decreasing order
    that add up to `n`.
    """
    if size == 1:
        yield [n]
        return
    if limit is None:
        limit = n
    start = (n + size - 1) // size
    stop = min(limit, n - size + 1) + 1
    for i in range(start, stop):
        for tail in _sum_to_n(n - i, size - 1, i):
            yield [i] + tail
            
def _generate_N_tuple_sum_S(N, S):
    sets = []
    for partition in _sum_to_n(S+N, N):
        sets.append(tuple(np.array(partition)-1))
    return sets

def _multinomial(lst):
    res, i = 1, sum(lst)
    i0 = lst.index(max(lst))
    for a in lst[:i0] + lst[i0+1:]:
        for j in range(1,a+1):
            res *= i
            res //= j
            i -= 1
    return res

def _tuple_multiplicity(tpl):
    elems_uniq, elems_count = np.unique(list(tpl), return_counts=True)
    return _multinomial(list(elems_count))

