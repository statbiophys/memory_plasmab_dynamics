import numpy as np
import pandas as pd
from os import listdir


def read_pars(path):
    f = open(path, 'r')
    pars = dict()
    for l in f:
        try:
            pars[l.split('\t')[0]] = float(l.split('\t')[1])
        except:
            pass
    f.close()
    return pars


def read_family_frame(sample_name, is_replicate=False, fam_type='familiy_pairs'):
    if is_replicate:
        f = pd.read_csv('sequences/replicates/'+sample_name+'.tsv', sep='\t', index_col=0, low_memory=False)
    else:
        f = pd.read_csv('sequences/'+sample_name+'.tsv', sep='\t', index_col=0, low_memory=False)
    f = f[f.chain == 'H']
    return f[f[fam_type].notna()]


def write_collapsed_ids(collapsed_ids, path):
    """
    The collapsed ids is the list of identifiers of sequences that have been collapsed together.
    Their knowledge will be useful, for example, to map back lineages in replicates
    """
    
    f = open(path, 'w')
    for row in collapsed_ids.items():
        out_list = ""
        for _id in row[1]:
            out_list += _id + ',' 
        if row[0] == row[1][0]:
            f.write(out_list[:-1] + '\n')
        else:
            print('first id in list is not the key')
    f.close()
    
    
def read_collapsed_ids(path):
    collapsed_ids = dict()
    f = open('lineages/src_data/pat1_hilary_heavy_ids.txt', 'r')
    for l in f.readlines():
        ids = l.split(',')
        ids[-1] = ids[-1][:-1]
        collapsed_ids[ids[0]] = ids
    f.close()
    return collapsed_ids


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


def _read_infer_result(f, result):
    
    read_params, read_errs = False, False
    for l in f.readlines():

        key = l.split('\t')[0][:-1]
        if read_params and not read_errs and len(l.split('\t')) > 1:
            result[key] = float(l.split('\t')[1])
        if read_errs and len(l.split('\t')) > 1:
            result['err_'+key] = float(l.split('\t')[1])

        if key == 'success': result[key] = l.split('\t')[1][:-1]
        if key == 'minus_ll': result['ll'] = -float(l.split('\t')[1])
        if key == 'best parameters:': read_params = True
        if key == 'errors:': read_errs = True
            
    return result



def read_noise_result(path, model):
    
    f = open(path + 'infer_noise_'+model+'.txt')
    result = dict()
    result['model'] = model
    result = _read_infer_result(f, result)
    f.close()
    return result


def read_gbm_result(path, n1_min, sub_name=""):
    
    full_path = path + 'infer_gbm_'+sub_name+'n1min_'+str(n1_min)+'.txt'
    #if is_seq: full_path = path + 'infer_gbm_seq_n1min_'+str(n1_min)+'.txt'
    f = open(full_path)
    result = dict()
    result['n1_min'] = n1_min
    result = _read_infer_result(f, result)
    f.close()
    return result


def import_sample_counts(sample_name, metadata):
    merged_counts = pd.DataFrame(columns=['familiy_pairs'])
    for r in range(metadata.loc[sample_name].repl_count):
        r_fr = read_family_frame(sample_name+'_r'+str(r+1), True, fam_type='familiy_pairs')
        f_aux = r_fr.groupby('familiy_pairs').agg({'pair_count':sum})
        #print('N cells repl ' + str(r+1) + ':', np.sum(f_aux.pair_count))
        merged_counts = pd.merge(merged_counts, f_aux, on='familiy_pairs', how='outer', suffixes=('', '_'+str(r+1))).fillna(0)
    count_mat = np.array(merged_counts.drop('familiy_pairs', axis=1).values, dtype=int)
    sp = build_sparse_counts(count_mat.T)
    n_uniq = sp[['n'+str(r+1) for r in range(len(count_mat[0]))]].values
    n_counts = sp['occ'].values
    return n_uniq, n_counts


def read_noise_result(path, model):
    
    f = open(path + 'infer_noise_'+model+'.txt')
    result = dict()
    result['model'] = model
    
    read_params, read_errs = False, False
    for l in f.readlines():

        key = l.split('\t')[0][:-1]
        if read_params and not read_errs and len(l.split('\t')) > 1:
            result[key] = float(l.split('\t')[1])
        if read_errs and len(l.split('\t')) > 1:
            result['err_'+key] = float(l.split('\t')[1])

        if key == 'success': result[key] = l.split('\t')[1][:-1]
        if key == 'minus_ll': result['ll'] = -float(l.split('\t')[1])
        if key == 'best parameters:': read_params = True
        if key == 'errors:': read_errs = True

    f.close()
    return result


def update_result_frame(frame, vals_i, vals_j, res_mat):
    
    R_old, R = 0, res_mat.shape[2]
    if len(frame) > 0: R_old = max(frame.index) + 1
        
    if R_old == 0:
        frame = pd.DataFrame(index=np.arange(R))
    else:
        for r in range(R):
            frame.loc[R_old+r] = [np.nan for _ in range(len(frame.columns))]
        
    for i, vi in enumerate(vals_i):
        for j, vj in enumerate(vals_j):
            pair = "%3.2f_%3.2f"%(vj, vi)
            frame.loc[np.arange(R_old, R_old + R), pair] = res_mat[i,j,:]
            
    return frame


def read_result_frame(frame):
    val_x = np.unique(frame.columns.str.split('_').str[0])
    val_x = val_x[np.argsort(np.array(val_x, dtype=float))]
    val_y = np.unique(frame.columns.str.split('_').str[1])
    val_y = val_y[np.argsort(np.array(val_y, dtype=float))]
    counts, means, stds = np.zeros((len(val_y), len(val_x))), np.zeros((len(val_y), len(val_x))), np.zeros((len(val_y), len(val_x)))
    for pair, vals in frame.T.iterrows():
        x = pair.split('_')[0]
        y = pair.split('_')[1]
        j = list(val_x).index(x)
        i = list(val_y).index(y)
        vals = vals.dropna()
        counts[i,j] = len(vals)
        means[i,j] = np.mean(vals)
        stds[i,j] = np.std(vals)
    return counts, means, stds, np.array(val_x, dtype=float), np.array(val_y, dtype=float)