import numpy as np
import matplotlib.pyplot as plt


def ellipses_path(center, semi_axis, rotation_angle):
    ts = np.linspace(0, np.pi*2, 200)
    cost, sint, cosa, sina = np.cos(ts), np.sin(ts), np.cos(rotation_angle), np.sin(rotation_angle)
    xs = semi_axis[0] * cosa * cost - semi_axis[1] * sina * sint + center[0]
    ys = semi_axis[0] * sina * cost + semi_axis[1] * cosa * sint + center[1]
    return xs, ys
    

def plot_scatter(ax, sparse_counts, samp_name1, samp_name2, min_s=10, max_s=500, legend=True):
    
    # Setting the zeros to half the minimal frequency
    N1 = np.sum(sparse_counts['n1'] * sparse_counts['occ'])
    freq1 = np.where(sparse_counts['freq1'] == 0, 0.5 / N1, sparse_counts['freq1'])
    N2 = np.sum(sparse_counts['n2'] * sparse_counts['occ'])
    freq2 = np.where(sparse_counts['freq2'] == 0, 0.5 / N2, sparse_counts['freq2'])
    
    # Setting the size of the dots depending on the occurrence
    max_log_o = np.ceil(np.log10(np.max(sparse_counts.occ)))
    norm_occ = np.log10(sparse_counts.occ) / max_log_o
    dot_size = norm_occ * (max_s - min_s) + min_s
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('freq. family in ' + samp_name1, fontsize=12)
    ax.set_ylabel('freq. family in ' + samp_name2, fontsize=12)
    ax.set_title('tot. counts: ' + str(int(N1)) + ', ' + str(int(N2)))
    
    ax.scatter(freq1, freq2, alpha=0.5, s=dot_size, c='b')
    
    for log_s in np.arange(max_log_o+1):
        ns = log_s / max_log_o
        ax.scatter([], [], s=ns * (max_s - min_s) + min_s, label='n. occurrences='+str(int(10**log_s)), c='b', alpha=0.5)
    if legend:
        ax.legend()
    
    x_th = min(freq1) * 1.33
    ys = np.linspace(min(freq2), max(freq2), 20)
    ax.plot(np.ones(len(ys))*x_th, ys, c='k')
    y_th = min(freq2) * 1.33
    xs = np.linspace(min(freq1), max(freq1), 20)
    ax.plot(xs, np.ones(len(xs))*y_th, c='k')
    xs = np.linspace(max(min(freq1), min(freq2)), min(max(freq1), max(freq2)), 20)
    ax.plot(xs, xs, c='k', ls='--')

    uni_fr = np.unique(freq1)
    asort = np.argsort(uni_fr)
    zero_f, min_f, max_f = uni_fr[asort[0]], uni_fr[asort[1]], uni_fr[asort[-1]]
    xticks = 10**np.arange(np.ceil(np.log10(min_f)), np.ceil(np.log10(max_f)))
    ax.set_xticks(np.append(zero_f, xticks))
    ax.set_xticklabels(["%.g"%x for x in np.append(0, xticks)])
    
    uni_fr = np.unique(freq2)
    asort = np.argsort(uni_fr)
    zero_f, min_f, max_f = uni_fr[asort[0]], uni_fr[asort[1]], uni_fr[asort[-1]]
    yticks = 10**np.arange(np.ceil(np.log10(min_f)), np.ceil(np.log10(max_f)))
    ax.set_yticks(np.append(zero_f, yticks))
    ax.set_yticklabels(["%.g"%x for x in np.append(0, yticks)])
    
    return ax


def map_imshow_axis(xs, values, flip=False):
    if flip:
        return len(xs)-1-(len(xs)-1)*(values - xs[0])/(xs[-1] - xs[0])
    else:
        return (len(xs)-1)*(values - xs[0])/(xs[-1] - xs[0])
    
    
def plot_ll_grid(ax, lls, x_par, y_par, delta_vmin=1, cb=True):
    
    ax.set_xlabel('sigma', fontsize=12)
    ax.set_ylabel('tau', fontsize=12)
    tics_step=int(len(lls)/5)
    ax.set_xticks(np.arange(0, len(x_par), tics_step))
    ax.set_xticklabels(["%3.2f"%s for s in x_par[::tics_step]])
    ax.set_yticks(np.arange(0, len(y_par), tics_step))
    ax.set_yticklabels(["%3.2f"%s for s in y_par[::-tics_step]])
    
    vmax = max(lls[lls != float('inf')])
    im = ax.imshow(lls[::-1], vmax=vmax, vmin=vmax-delta_vmin)
    
    i_max, j_max = np.unravel_index(np.argmax(lls, axis=None), lls.shape)
    ax.scatter([map_imshow_axis(x_par, x_par[j_max])], [map_imshow_axis(y_par, y_par[i_max], True)])
    
    if cb:
        plt.colorbar(im, ax=ax)
    return ax
