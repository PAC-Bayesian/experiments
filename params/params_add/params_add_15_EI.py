import numpy as np
import pickle

path_to_save = './params/params_add/'



kw_prior_1 = {'f_true': 'f_1d_single_steep', 
            'input_dim': 1,
            'noise_std': 0.2, 
            'ref': np.array([[0.01], [0.5], [0.95]]),
            'grid_size': (100, ), 
            'N_init': 8, 
            'bounds': ((0, 1), ), 
            'seed': 111
            }

kw_prior_2 = {'f_true': 'f_1d_double', 
            'input_dim': 1,
            'noise_std': 0.2, 
            'ref': np.array([[0.98]]),
            'grid_size': (100, ), 
            'N_init': 8, 
            'bounds': ((0, 1), ), 
            'seed': 111
            }

kw_prior_plot = {'path_figs': './figs/fig_add_15/',
                 'save': True, 
                 'title': None
                }

kw_list = ([kw_prior_1] * 2  + [kw_prior_2]) * 5
d_factor = len(kw_list)

kw_add_post={'kw_list': kw_list, 
             'num_comps': d_factor, 
             'noise_var': 1e-8}



kw_samplers = {'samplers': {'SGD_EI': 'TSSGD'},
               'args': {'SGD_EI': {'num_anim': 5,
                                   'N_new': 300,
                                   'details': ["EI"], 
                                   'search_step': "EI",
                                   'search_dir': "realized_GD", 
                                   'min_util': 1e-3, 
                                   'tol': 0.1
                                  }
                                }
              }

kw_run = {'path_data': './data/data_add_15/',
          'save_meta': True, 
          'path_models': './models/model_space_add_15/',
          'save_model': False }

kws = {'kw_prior_1': kw_prior_1, 
       'kw_prior_2': kw_prior_2, 
       'kw_prior_plot': kw_prior_plot, 
       'kw_samplers': kw_samplers, 
       'kw_run': kw_run, 
       'kw_add_post': kw_add_post    
       }

file_to_save = open(path_to_save + 'params_add_15_EI', 'wb')
pickle.dump(kws, file_to_save)
file_to_save.close()