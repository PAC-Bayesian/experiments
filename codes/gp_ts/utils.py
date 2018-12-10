import numpy as np
import inspect
import scipy as sp
import pickle
import builtins
import os
import GPy
import gc
import re


### helpers
def make_dirs(path):
    try:
        os.makedirs(path)
    except OSError:
        pass

def get_summary_stats(data):
    summary_funcs = {'avg': np.mean, 'median': np.median, 'std': np.std}
    return {k: v(data) for k, v in summary_funcs.items()}

def print_model(model, path='./', file_name='fitted_GP_prior'):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    with open(path + file_name, 'wt') as f:
        f.write(ansi_escape.sub('', model.__str__()))
    f.close()


### Grids transformation
def transform_grid_X(X, mesh=False):
    """
    Transform a sparse meshgrid X to data matrix of N * input_dim.
    """
    if mesh:
        X = np.meshgrid(*X, indexing='ij')
    X_to_flatten = [x.flatten()[:,np.newaxis] for x in X]
    return np.hstack(X_to_flatten)

def transform_grid_Y(Y):
    return Y.flatten()[:, np.newaxis]


### Offset Woodbury matrix
def offset_woodbury(model_seq, num_data_init):  
    """
    Offset the automatically added noise variance to the Woodbury matrix by GPy
    if the target is the latent function value f.
    """ 
    ind_diag = (np.arange(num_data_init, model_seq.X.shape[0]), ) * 2
    var = model_seq.likelihood.gaussian_variance().values
    model_seq.posterior._K[ind_diag] -= var
    model_seq.posterior._K[np.diag_indices(model_seq.X.shape[0])] -= (1 - 1e-2) * 1e-8

    post, *_ = model_seq.inference_method.inference(kern=model_seq.kern, \
    X=model_seq.X, Y=model_seq.Y, \
    likelihood=model_seq.likelihood, \
    K=model_seq.posterior._K)
    model_seq.posterior = post


### Save and load data
def draw_and_save(samplers, target='f', cond=False, verbose=True, \
    num_sims=1, path='./data/', save_meta=True, **kw):
    """Save value_min and x_argmin of each sampler."""
    seed = kw.get('seed', 111)

    value_min = dict()
    x_argmin = dict()
    meta_data = dict()
    random_exp = kw.get('random_exp', False)
    for n in range(num_sims):
        print('round: {0}'.format(n))
        for s_name in samplers:
            s = samplers[s_name]
            print(s)
            if save_meta:
                meta_data['{0}'.format(s.name)] = {k:v for k, v in s.__dict__.items() \
                if type(v).__name__ in dir(builtins) and 'info' not in k and 'space' not in k}
            if 'TSSGD' in str(type(s)):            
                s.draw_samples(verbose=verbose, random_exp=random_exp, seed=seed)
            else:
                s.draw_samples(seed=seed)
            
            s.summarize_samples()
            value_min['{0}'.format(s.name)] = s.value_min
            x_argmin['{0}'.format(s.name)] = s.x_argmin
            if hasattr(s, 'num_evals') and save_meta:
                meta_data['{0}'.format(s.name)]['num_evals'] = get_summary_stats(s.num_evals)

        file_to_save = open(path + '{0}_{1}'.format(target, n), 'wb')
        pickle.dump({'value_min':value_min, 'x_argmin':x_argmin, 'meta_data': meta_data}, file_to_save)
        file_to_save.close()
        gc.collect()

def load_and_pack(num, target='f', path='./data/'):
    """Load samples and pack them for comparison.
    : param num: range or list
            data indices to include
    """
    value_min = dict()
    x_argmin = dict()
    for n in num:
        file_to_load = open(path + '{0}_{1}'.format(target, n), 'rb')
        data = pickle.load(file_to_load)
        assert data['value_min'].keys() == data['x_argmin'].keys()
        keys = data['value_min'].keys()
        for k in keys:
            value_min[k] = np.hstack((value_min.get(k, np.tile([], data['value_min'][k].shape )), data['value_min'][k]))
            x_argmin[k] = np.hstack((x_argmin.get(k,np.tile([], data['x_argmin'][k].shape)), data['x_argmin'][k]))

        file_to_load.close()
    return value_min, x_argmin


### Save and load models
def save_model_space(path_to, model_space):
    """Save a list of list of models.
    """
    for i, traj in enumerate(model_space):
        path_traj = path_to + 'traj_{}/'.format(i)
        make_dirs(path_traj)
        for k, m in enumerate(traj):
            m.save_model(path_traj + 'm_{}'.format(k)) 

def load_model_space(path_from):
    trajs = os.listdir(path_from)
    model_space = [[] for i in range(len(trajs))]
    for i in range(len(trajs)):
        path_traj = path_from + 'traj_{}/'.format(i)
        for k in range(len(os.listdir(path_traj))):
            m = GPy.Model.load_model(path_traj + 'm_{}.zip'.format(k))
            model_space[i].append(m)
    return model_space
    