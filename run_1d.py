import pickle
import argparse

import codes.gp_ts
import codes.plotting
from codes.plotting import FIG_SIZE, FIG_SIZE_SUB2

from codes.gp_ts.prepare import Prior
from codes.gp_ts.utils import *
from codes.gp_ts.funcs import * 
from codes.plotting.prior_plots import PriorPlots 
from codes.gp_ts.ts_grid import TSGridJoint
from codes.gp_ts.ts_SGD import TSSGD
from codes.gp_ts.ts_bochner import TSBochnerGrid, TSBochnerOpt 


## parser arguments
parser = argparse.ArgumentParser(description='Input number of samples and number of .')
parser.add_argument("path_to_params", type=str, help='path to the parameters file')
parser.add_argument("--num_samples", dest="num_samples", type=int, default=100)
parser.add_argument("--num_sims", dest="num_sims", type=int, default=10)

args = parser.parse_args()

num_samples = args.num_samples
num_sims = args.num_sims
path_to_params = args.path_to_params

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

path_to = kw_prior_plot['path_figs'] + 'num_samples-{}/'.format(num_samples)
make_dirs(path_to)


def gen_prior(**kw):
    seed = kw.get('seed', 1111)
    input_dim = kw.get('input_dim', 1)
    f_true = eval(kw.get('f_true', 'f_1d_forrester'))
    bounds = kw.get('bounds', bounds_1d)
    noise_std = kw.get('noise_std', 0.05)
    N_init = kw.get('N_init', 8)
    grid_size = kw.get('grid_size', (50,))
    ref = kw.get('ref', None)
    
    prior = Prior(f_true=f_true, bounds=bounds, seed=seed, input_dim=input_dim)
    prior.noise_f(noise_std=noise_std) 
    prior.gen_data_init(N_init=N_init,ref=ref) 
    prior.gen_grid(grid_size=grid_size)
    prior.fit_GP_model()
    
    return prior

def plot_prior(prior, **kw):
    path_figs = kw.get('path_figs', './figs/')
    make_dirs(path_figs)
    save = kw.get('save', False)
    title = kw.get('title', None)
    
    plots = PriorPlots(prior, save=save, path=path_figs, figsize=FIG_SIZE)
    plots.plot_f_true(title=title)
    plots.plot_model(title=title)
    if prior.input_dim == 2:
        plots.plot_contour_2d(figsize=FIG_SIZE_SUB2, latent=True)
        plots.plot_contour_2d(figsize=FIG_SIZE_SUB2, latent=False, scatter=True)

def gen_samplers(prior, **kw):
    samplers = {}
    for n, c in kw_samplers['samplers'].items():
        s = eval(c)(prior=prior, num_samples=num_samples, name=n, \
                    **kw_samplers['args'][n])
        samplers[n] = s
    return samplers 

def run_samplers(samplers, **kw):
    path_data = kw.get('path_data', './data/')
    path_data += 'num_samples-{}/'.format(num_samples)
    save_meta = kw.get('save_meta', False)
    path_models = kw.get('path_models', './models/')
    make_dirs(path=path_data)
    draw_and_save(samplers=samplers, path=path_data, save_meta=save_meta, num_sims=num_sims)
    for s_name in samplers:
        s = samplers[s_name]
        if isinstance(s, TSSGD):
            make_dirs(path_models + '{}/'.format(s_name))
            save_model_space(path_to=path_models + '{}/'.format(s_name), model_space=s.model_space)

def main():
    prior = gen_prior(**kw_prior)
    plot_prior(prior, **kw_prior_plot)
    samplers = gen_samplers(prior, **kw_samplers)
    print(samplers)
    run_samplers(samplers, **kw_run)
    samplers['Bochner_opt'].show_GP_sample(ind=[0], save=True, path_to=path_to)
    samplers['Bochner_opt'].show_GP_sample(ind=[1], save=True, path_to=path_to)
    samplers['Bochner_opt'].show_GP_sample(ind=[10], save=True, path_to=path_to)
    samplers['Bochner_opt'].show_GP_sample(ind=[11], save=True, path_to=path_to)

if __name__ == "__main__":
    main() 
