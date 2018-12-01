import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import seaborn as sns
import pandas as pd
from ..gp_ts.utils import load_and_pack, offset_woodbury
from .stat_helper import contrast, calculate_hist
import scipy as sp
from scipy import stats
import GPy


def contrast_and_boxplot(num, input_dim, target='f', bins=20, path_from='./data/', figsize=(15,5), \
save=False, path_to='./', ext='.pdf', file_name='box_plot-Hellinger_dist', label_dict=None, benchmark=None, **kw):

    set_title = kw.get("set_title", False)
    if isinstance(num, int):
        num=range(num)
    file_name += path_from[path_from.rfind('/')+1:] + '-input_dim-{}'.format(input_dim)

    value_min, x_argmin = load_and_pack(target=target, num=num, path=path_from)
    if label_dict:
        value_min = {k: value_min[k] for k in label_dict}
        x_argmin = {k: x_argmin[k] for k in label_dict}
    dist_value_min = pd.DataFrame.from_dict(contrast(value_min, bins=bins, obj='value_min', \
    label_dict=label_dict, benchmark=benchmark))
    dist_x_argmin = pd.DataFrame.from_dict(contrast(x_argmin, bins=bins, obj='x_argmin', input_dim=input_dim, \
    label_dict=label_dict, benchmark=benchmark))

    plt.figure(figsize=figsize)
    # dist_x_argmin.boxplot()
    sns.boxplot(data=dist_x_argmin)
    plt.xlabel('sampler pair')
    plt.ylabel('Hellinger distance')
    if set_title:
        plt.title('x_argmin Hellinger_dist boxP (input_dim={0}, num_sims={1})'.format(input_dim, len(num)))
    plt.tight_layout() 
    if save: 
        plt.savefig(path_to + file_name + '-x_argmin' + ext)
    plt.clf()
    plt.close()

    plt.figure(figsize=figsize)
    # dist_value_min.boxplot()
    sns.boxplot(data=dist_value_min)
    plt.xlabel('sampler pair')
    plt.ylabel('Hellinger distance')
    if set_title:
        plt.title('value_min Hellinger_dist boxP (target={0}, num_sims={1})'.format(target, len(num)))
    plt.tight_layout()
    if save: 
        plt.savefig(path_to + file_name + '-value_min' + ext)
    plt.clf()
    plt.close()
    return dist_x_argmin, dist_value_min

def traj_plot_1d(traj, save=False, figsize=None, path_to='./', **kw):
    color = kw.get('color', 'red')
    assert traj, "empty sequence of models"
    num_data_init = traj[0].num_data
    bounds = kw.get("bounds", (0, 1))
    for k, m in enumerate(traj[1:]):
        m = m.copy()
        offset_woodbury(m, num_data_init)
        fig = GPy.plotting.plotting_library().figure(figsize=figsize)
        m.plot_f(figure=fig, plot_limits=bounds)
        m.plot_data(which_data_rows=range(num_data_init, m.num_data-1), figure=fig)
        m.plot_data(which_data_rows=range(m.num_data-1, m.num_data), color=color, figure=fig)
        if save:
            plt.savefig(path_to + 'm_{}'.format(k))
        plt.clf()
        plt.close()

def pmf_plot(p, q, label=None, margin=0.01, figsize=None, **kw):
    """Bar plot of two pmf: p and q."""
    assert len(p) == len(q), "p and q must have the same length"
    
    x = np.arange(len(p))
    label = ["KL(p|q): {:.3f}".format(stats.entropy(p, q)), "KL(q|p): {:.3f}".format(stats.entropy(q, p))]

    rv_p = stats.rv_discrete(name='rv_p', values=(x, p))
    rv_q = stats.rv_discrete(name='rv_q', values=(x, q))

    plt.figure(figsize=figsize)
    plt.plot(x - margin, rv_p.pmf(x), 'ro', label=label[0])
    plt.plot(x + margin, rv_q.pmf(x), 'bs', label=label[1])
    plt.vlines(x - margin, 0, rv_p.pmf(x), colors='r', linestyles='-')
    plt.vlines(x + margin, 0, rv_q.pmf(x), colors='b', linestyles='--')
    plt.legend()
    plt.title('cluster density')
    plt.ylabel('Probability')
    plt.show()


    

    