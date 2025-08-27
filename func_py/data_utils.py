import numpy as np
import pandas as pd
from os import listdir, path



def read_pars(path):
    f = open(path, 'r')
    pars = dict()
    for l in f:
        try:
            key = l.split('\t')[0]
            if key[-1] == ':':
                key = key[:-1]
            pars[key] = float(l.split('\t')[1])
        except:
            pass
    f.close()
    return pars


def read_family_frame(sample_name, is_replicate=False, fam_type='familiy_pairs'):
    if is_replicate:
        f = pd.read_csv('sequences/replicates/'+sample_name+'.tsv', sep='\t', index_col=0, low_memory=False)
    else:
        f = pd.read_csv('sequences/'+sample_name+'.tsv', sep='\t', index_col=0, low_memory=False)
    if 'chain' in f.columns:
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


def import_and_build_sparse_counts(sample_list, fam_label='familiy_pairs', count_label='pair_count'):
    aux = pd.DataFrame()
    for sample_name in sample_list:
        fr = read_family_frame(sample_name, fam_type=fam_label)
        if count_label not in fr.columns:
            print('count label not found, setting the counts to 1')
            fr[count_label] = 1
        clone_counts = fr.groupby(fam_label).agg({count_label : sum})
        clone_counts = clone_counts.rename({count_label : 'counts_' + sample_name}, axis=1)
        aux = pd.merge(aux, clone_counts, how='outer', left_index=True, right_index=True).fillna(0)
    return build_sparse_counts(np.array(aux.astype(int)).T)


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
    
    #f = open(path + 'infer_noise_'+model+'.txt')
    f = open(path)
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


def import_sample_counts(sample_name, n_repl, family_label='familiy_pairs', count_label='pair_count'):
    """
    For a given sample with multiple replicate it returns the sparse representation
    of counts across replicates: a list of unique counts per replicate [n_1, n_2, ..., n_R]
    and the multiplicity of that configuration.
    """
    merged_counts = pd.DataFrame(columns=[family_label])
    
    for r in range(n_repl):
        r_fr = read_family_frame(sample_name+'_r'+str(r+1), True, fam_type=family_label)
        if count_label not in r_fr.columns:
            r_fr[count_label] = 1
        f_aux = r_fr.groupby(family_label).agg({count_label:sum})
        merged_counts = pd.merge(merged_counts, f_aux, on=family_label, how='outer', 
                                 suffixes=('', '_'+str(r+1))).fillna(0)
    count_mat = np.array(merged_counts.drop(family_label, axis=1).values, dtype=int)
    sp = build_sparse_counts(count_mat.T)
    n_uniq = sp[['n'+str(r+1) for r in range(len(count_mat[0]))]].values
    n_counts = sp['occ'].values
    
    return n_uniq, n_counts


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


def downsample_frame(f, n_reads, count_label='pair_count'):
    
    all_seqs = np.array([])
    n_counts = np.unique(f[count_label].values)
    for n in n_counts:
        ids = np.array(list(f[f[count_label] == n].index))
        ids = np.repeat(ids, n)
        all_seqs = np.append(all_seqs, ids)
    
    sub_samples = np.random.choice(all_seqs, n_reads, replace=False)
    uni_subsamp, count_subsamp = np.unique(sub_samples, return_counts=True)
    sub_fr = f.loc[uni_subsamp]
    sub_fr[count_label] = count_subsamp
    return sub_fr


def discretize(xs_list, n_discr):
    """
    Discretization of the values in xs_list in n_discr bins. xs list is a 2d array.
    It returns the count of xs in each bin and the binned interval xs
    """
    x_bins = np.linspace(np.min(xs_list), np.max(xs_list), n_discr)
    x_vals = x_bins + (x_bins[1] - x_bins[0]) / 2.0
    xs_bins = np.array([np.digitize(x, x_bins) for x in xs_list]) - 1
    return xs_bins, x_vals


def read_noise_negbin_a(folder, samples):
    
    noise_inferred = np.zeros(len(samples), dtype=bool)
    a_neg_bin = np.zeros(len(samples))
    for i, samp in enumerate(samples):
        pth = folder + samp + '_negbin.txt'
        isthere = path.exists(pth)
        if isthere:
            res = read_noise_result(pth, 'negbin')
            a_neg_bin[i] = res['a']
            noise_inferred[i] = True

    if not any(noise_inferred):
        print('noise model not found for any of the samples')

    if not all(noise_inferred):
        print('one or more samples do not have a lerned model, the parameter is set as the average of the others')
        a_mean = np.mean(a_neg_bin[noise_inferred])
        a_neg_bin[np.logical_not(noise_inferred)] = a_mean
        
    return a_neg_bin