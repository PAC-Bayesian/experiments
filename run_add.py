import pickle
import argparse
import gc

import codes.gp_ts
import codes.plotting
from codes.plotting import FIG_SIZE_1D, FIG_SIZE_1D_EXT, FIG_SIZE_2D, FIG_SIZE_2D_SUB2

from codes.gp_ts.prepare import Prior, Post, AdditivePrior, AdditivePost
from codes.gp_ts.utils import *
from codes.gp_ts.funcs import * 
from codes.plotting.prior_plot import PriorPlot 
from codes.gp_ts.ts_grid import TSGridJoint
from codes.gp_ts.ts_SGD import TSSGD



## parser arguments
parser = argparse.ArgumentParser(description='Input number of samples and number of .')
parser.add_argument("path_to_params", type=str, help='path to the parameters file')
parser.add_argument("--num_samples", dest="num_samples", type=int, default=100)
parser.add_argument("--num_sims", dest="num_sims", type=int, default=10)
parser.add_argument("--seed", dest="seed", type=int, default=1111)
args = parser.parse_args()

num_samples = args.num_samples
num_sims = args.num_sims
seed = args.seed
path_to_params = args.path_to_params

## load kws
params_to_load = open(path_to_params, 'rb')
kws = pickle.load(params_to_load) 
params_to_load.close()


kw_prior_1 = kws['kw_prior_1']
kw_prior_2 = kws['kw_prior_2']
kw_prior_plot = kws['kw_prior_plot']
kw_samplers = kws['kw_samplers']
kw_run = kws['kw_run']
kw_add_post = kws['kw_add_post'] 


path_to = kw_prior_plot['path_figs'] + 'num_samples-{}/'.format(num_samples)
make_dirs(path_to)

def gen_prior(**kw):
    seed = kw.get('seed', 1111)
    input_dim = kw.get('input_dim', 1)
    f_true = eval(kw.get('f_true', 'f_1d_forrester' if input_dim == 1 else 'f_2d_single'))
    bounds = kw.get('bounds', bounds_1d if input_dim == 1 else bounds_2d)
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
    
    figsize=FIG_SIZE_1D if prior.input_dim == 1 or isinstance(prior, AdditivePrior) else FIG_SIZE_2D
    plots = PriorPlot(prior, save=save, path=path_figs, figsize=figsize)
    
    if isinstance(prior, AdditivePrior):
        plots.plot_f_true_comp(title=title)
        plots.plot_model_comp(title=title)      
    else:
        plots.plot_f_true(title=title)
        plots.plot_model(title=title)
        if prior.input_dim == 2:
            plots.plot_contour_2d(figsize=FIG_SIZE_2D_SUB2, latent=True, scatter=False)
            plots.plot_contour_2d(figsize=FIG_SIZE_2D_SUB2, latent=False, scatter=True)

def gen_add_post(**kw):
    kw_list = kw.get('kw_list', [kw_prior_1])
    num_comps = kw.get('num_comps', 1)
    noise_var = kw.get('noise_var', 1e-8)
    assert len(kw_list) == kw['num_comps'] or len(kw_list) == 1
    if len(kw_list) == 1:
        kw_list *= num_comps
    priors = []
    for kw_prior in kw_list:
        p = gen_prior(**kw_prior)
        priors.append(p)
    post_add = AdditivePost(priors)
    post_add.fit_GP_model(noise_var=noise_var, verbose=False)
        
    return post_add  

def draw_samples_grid_comp(priors, num_samples=100, num_sims=1, seed=111, **kw):
    path = kw.get('path_data','./data/') + 'comp/'
    make_dirs(path=path)
    np.random.seed(seed)
    num_comps = kw_add_post.get('num_comps', 1)
    assert num_comps == len(priors)
    for i in range(num_sims):     
        samplers = []
        x_argmin = []
        value_min = []
        for k in range(num_comps):
            samplers.append(TSGridJoint(priors[k], num_samples, name='grid_comp_' + str(k)))   
        for s in samplers:
            s.draw_samples()
            s.summarize_samples()
            x_argmin.append(s.x_argmin)
            value_min.append(s.value_min)
        file_to_save = open(path + 'f_{}'.format(i), 'wb')
        pickle.dump({'value_min': {'joint_comp_{}'.format(num_comps): sum(value_min)}, 
                     'x_argmin': {'joint_comp_{}'.format(num_comps): np.hstack(x_argmin)}}, 
                   file_to_save)
        file_to_save.close()
        gc.collect()
    return samplers

def gen_samplers(prior, **kw):
    samplers = {}
    for n, c in kw_samplers['samplers'].items():
        s = eval(c)(prior=prior, num_samples=num_samples, name=n, \
                    **kw_samplers['args'][n])
        samplers[n] = s
    return samplers      

def run_samplers(samplers, seed=111, **kw):
    path_data = kw.get('path_data', './data/')
    path_data += 'num_samples-{}/'.format(num_samples)
    save_meta = kw.get('save_meta', False)
    save_model = kw.get('save_model', True)
    path_models = kw.get('path_models', './models/')
    make_dirs(path=path_data)
    draw_and_save(samplers=samplers, path=path_data, save_meta=save_meta, num_sims=num_sims, seed=seed)
    if save_model:
        for s_name in samplers:
            s = samplers[s_name]
            if isinstance(s, TSSGD):
                save_model_space(path_to=path_models, model_space=s.model_space)

def main():
    post_add = gen_add_post(**kw_add_post)
    plot_prior(post_add, **kw_prior_plot)
    draw_samples_grid_comp(post_add.priors, num_samples=num_samples, num_sims=num_sims, seed=seed, **kw_run)
    samplers = gen_samplers(post_add, **kw_samplers)
    print(samplers)
    run_samplers(samplers, seed=seed, **kw_run)

if __name__ == "__main__":
    main() 