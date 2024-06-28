import numpy as np
import pandas as pd
from os import listdir


def read_family_frame(sample_name, is_replicate=False, fam_type='familiy_pairs'):
    if is_replicate:
        f = pd.read_csv('sequences/replicates/'+sample_name+'.tsv', sep='\t', index_col=0, low_memory=False)
    else:
        f = pd.read_csv('sequences/'+sample_name+'.tsv', sep='\t', index_col=0, low_memory=False)
    f = f[f.chain == 'H']
    return f[f[fam_type].notna()]


def build_sparse_counts(count_list):
    """
    Sparse representation of counts of R replicates. Each unique combination of 
    R counts is associated with its multiplicity/occurrence in the replicates
    """
    
    count_fr = pd.DataFrame()
    Ns = [np.sum(counts) for counts in count_list]
    is_no_zero_mask = np.zeros(len(count_list[0]), dtype=bool)
    for i, counts in enumerate(count_list):
        count_fr['n'+str(i+1)] = np.array(counts, dtype=int)
        count_fr['n'+str(i+1)+'_str'] = np.array(counts, dtype=str)
        is_no_zero_mask = np.logical_or(is_no_zero_mask, np.array(counts)>0)
    count_fr = count_fr[is_no_zero_mask]
    
    count_fr['index'] = count_fr['n1_str']
    for i in range(1, len(count_list)):
        count_fr['index'] += '_' + count_fr['n'+str(i+1)+'_str']
        
    count_fr['occ'] = 1
    agg_dict = {'n'+str(i+1) : 'first' for i in range(len(count_list))}
    agg_dict['occ'] = sum   
    count_fr = count_fr.groupby('index').agg(agg_dict)

    for i in range(len(count_list)):
        count_fr['freq'+str(i+1)] = count_fr['n'+str(i+1)] / Ns[i]
    
    return count_fr.sort_values('occ', ascending=False)


def build_sparse_sort_counts(count_list):
    """
    Sparse-sort representation of counts of R replicates. Each unique combination of 
    R counts *without order* is associated with its multiplicity/occurrence in the replicates
    """
    
    sparse_sort_counts = build_sparse_counts(count_list)
    n_dim = len(count_list)
    
    ns_labels = ['n'+str(i) for i in range(1, n_dim+1)]
    ns = sparse_sort_counts[ns_labels].values
    a_sort = np.argsort(ns, axis=1)
    ns_sorted = np.take_along_axis(ns, a_sort, 1)
    sparse_sort_counts[ns_labels] = ns_sorted
    
    for n_lab in ns_labels:
        sparse_sort_counts[n_lab+'_str'] = np.array(sparse_sort_counts[n_lab], dtype=str)
    for n_lab in ns_labels[1:]:
        sparse_sort_counts['n1_str'] += '_' + sparse_sort_counts[n_lab+'_str']
    
    agg_dict = {n_lab : 'first' for n_lab in ns_labels}
    agg_dict['occ'] = sum    
    
    return sparse_sort_counts.groupby('n1_str').agg(agg_dict).sort_values('occ', ascending=False)


def get_cum_counts(values):
    """
    Compute the sorted cumulative counts of a list of values
    """
    uni_val, count_val = np.unique(values, return_counts=True)
    uni_val_s = uni_val[np.argsort(uni_val)[::-1]]
    count_val_s = count_val[np.argsort(uni_val)[::-1]]
    return uni_val_s[::-1], np.cumsum(count_val_s)[::-1]


def downsample_frame(f, n_reads):
    
    all_seqs = np.array([])
    n_counts = np.unique(f.pair_count.values)
    for n in n_counts:
        ids = np.array(list(f[f.pair_count == n].index))
        ids = np.repeat(ids, n)
        all_seqs = np.append(all_seqs, ids)
    
    sub_samples = np.random.choice(all_seqs, n_reads, replace=False)
    uni_subsamp, count_subsamp = np.unique(sub_samples, return_counts=True)
    sub_fr = f.loc[uni_subsamp]
    sub_fr.pair_count = count_subsamp
    return sub_fr