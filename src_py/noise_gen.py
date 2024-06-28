import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, nbinom
import sys
sys.path.insert(0, "../func_build/")
import like_func as lf
import my_utils as ut
from scipy.optimize import brentq, bisect


"""
Classes to generate artificial datasets generated from sampling of power laws.
"""


class Noise_Gen_Poiss : 
    """
    Generate artificial datasets generated from poisson sampling of a power law.
    """
    
    def __init__(self, alpha, N_seqs, N_discr_freq=int(1e4), verbose=False):
        
        self.alpha = alpha
        self.Ns = N_seqs
        self.verbose = verbose
        
        self.fmin = self._find_fmin_norm_freqs()
        if self.verbose:
            print('fmin fixed from normalization:', self.fmin)
            
        self.fs, self.probs = ut.generate_plaw_dist_log(alpha, self.fmin, N_discr_freq)
        self.f_samples = []
    
    
    def sample_repertoire(self):
        """
        Generate a sample of N_seqs from a power law distribution normalized on average. 
        """
        self.f_samples = np.random.choice(self.fs, self.Ns, p=self.probs)
        if self.verbose:
            print('The frequencies of the repertoire have been sampled')
        return self.f_samples
        
        
    def sample(self, N_reads, n_threads=8):
        """
        Generate a sample of N_reads on average, from a set of true frequencies.
        """
        if len(self.f_samples) == 0:
            self.sample_repertoire()
        return self._sampling_type(N_reads, n_threads)
        
        
    def _sampling_type(self, N_reads, n_threads):
        args = [(f, N_reads) for f in self.f_samples]
        return ut.parallelize_async(_aux_parall_sampler_poiss, args, n_threads)
    
    
    def _find_fmin_norm_freqs(self):
        """
        Find the value of fmin such that N_freqs form a power law distribution with exponent -alpha
        are normalized on average, i.e. <f> N = 1
        """
        f = lambda fmin : lf.plaw_average(self.alpha, np.exp(fmin)) * self.Ns - 1
        return np.exp(brentq(f, -np.log(self.Ns)-9, -np.log(self.Ns)+3, xtol=1e-16))
    

# For negative binomial we use the poisson class as parent class to derive 
# everything but the sampling function
class Noise_Gen_Negbin(Noise_Gen_Poiss) :
    """
    Generate artificial datasets generated from poisson negative binomial of a power law.
    """
    
    def __init__(self, alpha, N_seqs, a_nb, b_nb, N_discr_freq=int(1e4), verbose=False):
        self.a = a_nb
        self.b = b_nb
        super().__init__(alpha, N_seqs, N_discr_freq, verbose)
        
        
    def _sampling_type(self, N_reads, n_threads=8):
        args = [(f, N_reads, self.a, self.b) for f in self.f_samples]
        return ut.parallelize_async(_aux_parall_sampler_negbin, args, n_threads)
    
    
    
# AUXILIARY FUNCTIONS FOR PARALLEL COMPUTING

def _aux_parall_sampler_poiss(args):
    """
    One poisson sampling. Function passed to a multi-thread function
    """
    f, n_reads = args
    return poisson(n_reads * f).rvs()


def _aux_parall_sampler_negbin(args):
    """
    One negative binomial sampling. Function passed to a multi-thread function
    """
    f, n_reads, a, b = args
    mean = f * n_reads
    var = mean + a * mean ** b
    return nbinom(mean*mean/(var-mean), mean/var).rvs()



