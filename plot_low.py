import argparse
import pickle
import json
import warnings
import codes.gp_ts
from codes.gp_ts.utils import *
import codes.plotting
from codes.plotting import FIG_SIZE_1D, FIG_SIZE_1D_EXT, FIG_SIZE_2D, FIG_SIZE_2D_SUB2
from codes.plotting.plot_helpers import contrast_and_boxplot, traj_plot_1d, traj_plot_2d
from codes.plotting.samples_plot import SamplesPlot

warnings.filterwarnings("ignore")

## parser arguments
parser = argparse.ArgumentParser(description='Input number of samples and number of .')
parser.add_argument("path_to_params", type=str, help='path to the parameters file')
parser.add_argument("--num_samples", dest="num_samples", type=int, default=100)
parser.add_argument("--num_sims", dest="num_sims", type=int, default=10)
parser.add_argument("--ind_sim", dest="ind_sim", type=int, default=0)
parser.add_argument("--ind_traj", dest="ind_traj", type=int, default=0)


args = parser.parse_args()

num_samples = args.num_samples
num_sims = args.num_sims
path_to_params = args.path_to_params
ind_sim = args.ind_sim
ind_traj = args.ind_traj

## load kws
params_to_load = open(path_to_params, 'rb')
kws = pickle.load(params_to_load) 
params_to_load.close()

kw_prior = kws['kw_prior']
kw_prior_plot = kws['kw_prior_plot']
kw_samplers = kws['kw_samplers']
kw_run = kws['kw_run']
kw_plot_box = kws['kw_plot_box']
kw_plot_samples = kws['kw_plot_samples']
kw_plot_traj = kws['kw_plot_traj']

path_from = kw_run['path_data'] + 'num_samples-{}/'.format(num_samples)
path_to = kw_prior_plot['path_figs'] + 'num_samples-{}/'.format(num_samples)

def plot_box(**kw):
    save = kw.get('save', True)
    benchmark = kw.get('benchmark', None)
    bins = kw.get('bins', 20)
    label_dict = kw.get('label_dict', None)
    figsize = list(np.array(FIG_SIZE_1D) * (len(kw_samplers['samplers']) * 0.375))
    make_dirs(path_to)
    contrast_and_boxplot(num=num_sims, input_dim=kw_prior['input_dim'], benchmark=benchmark, bins=bins, 
                        path_from=path_from, path_to=path_to, 
                        figsize=figsize, save=save, label_dict=label_dict)

def plot_samples(ind_sim=0, **kw):
    save = kw.get('save', True)
    bins = kw.get('bins', 25)
    path_from_ind = path_from + '{0}_{1}'.format('f', ind_sim)
    path_to_ind = path_to + '{0}_{1}/'.format('f', ind_sim)
    make_dirs(path_to_ind)
    for label in kw.get('labels'):
        plots = SamplesPlot(label=label, save=save,  
                             figsize=FIG_SIZE_1D,
                             path_from=path_from_ind, 
                             path_to=path_to_ind)
        if np.all(np.array(plots.samples_input_dim) == 1):
            plots.contrast_hist_margin_1d(bins=bins)
        elif np.all(np.array(plots.samples_input_dim) == 2):
            plots.contrast_hist_margin_2d(bins=bins, figsize=FIG_SIZE_2D_SUB2, KDE=False)
            plots.contrast_hist_margin_1d(bins=bins)

def plot_traj(ind_traj=0, is_print=False, path_info=kw_prior_plot['path_figs'], **kw):
    save = kw.get('save', True)
    s_names = kw.get('s_names', ['SGD_EI'])
    
    for name in s_names:
        path_to_model = kw_run.get('path_models', './models/') + '{}/'.format(name)
        model_space = load_model_space(path_to_model)
        bounds = tuple(zip(*kw_prior['bounds']))
        path_to_ind = path_to + '{}_'.format(name) + '{0}_{1}/'.format('traj', ind_traj)
        make_dirs(path_to_ind)
        if kw_prior['input_dim'] == 1:
            traj_plot_1d(model_space[ind_traj], figsize=FIG_SIZE_1D, path_to=path_to_ind, save=save, bounds=bounds, 
            is_print=is_print, path_info=path_info)
        else:
            traj_plot_2d(model_space[ind_traj], figsize=FIG_SIZE_2D_SUB2, path_to=path_to_ind, save=save, bounds=bounds)

def save_num_evals(s_names=['SGD_EI'], ind_sim=0):
    file_to_load = open(path_from + '{0}_{1}'.format('f', 0), 'rb')
    data = pickle.load(file_to_load)
    file_to_load.close()
    for name in s_names:
        path_to_save = path_from + 'meta/'
        make_dirs(path_to_save)
        file_to_save = open(path_to_save + '{}-'.format(name) + 'num_evals_{}'.format(ind_sim), 'w')
        json.dump(data['meta_data'][name]['num_evals'], file_to_save)
        file_to_save.close()

def main():
    plot_box(**kw_plot_box)
    plot_samples(ind_sim=ind_sim, **kw_plot_samples)
    plot_traj(ind_traj=ind_traj, **kw_plot_traj)
    save_num_evals(s_names=kw_plot_traj['s_names'], ind_sim=ind_sim)

if __name__ == "__main__":
    main() 
