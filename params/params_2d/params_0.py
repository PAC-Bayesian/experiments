import numpy as np
import pickle

path_to_save = './params/params_2d/'

kw_prior = {'f_true': 'f_2d_single', 
            'input_dim': 2,
            'noise_std': 0.2,  
            'grid_size': (50, 50), 
            'N_init': 6, 
            'ref': np.array([[0.025, 0.025], [0.5, 0.5]]),
            'bounds': ((0, 1), (0, 1))
            }

kw_prior_plot = {'path_figs': './figs/fig_2d_0/', \
                 'save': True, \
                 'title': None,
                }

kw_samplers = {'samplers': {'joint_1': 'TSGridJoint', 
                            'joint_2': 'TSGridJoint',
                            'Bochner_grid': 'TSBochnerGrid',
                            'Bochner_opt': 'TSBochnerOpt',
                            'SGD_EI': 'TSSGD', 
                            'SGD_LCB':'TSSGD'                          
                           },
               'args': {'SGD_EI': {'num_anim': 50,
                                   'N_new': 10,
                                   'details': ["EI"], 
                                   'search_step': "EI",
                                   'search_dir': "realized_GD", 
                                   'min_util': 5e-4, 
                                   'tol': 0.05 
                                  },
                        'SGD_LCB': {'num_anim': 50,
                                     'N_new': 10,
                                     'details': ["LCB"], 
                                     'search_step': "LCB",
                                     'search_dir': "realized_GD", 
                                     'min_util': 5e-4, 
                                     'tol': 0.05, 
                                     'kappa': 3.0
                                    },
                        'joint_1': {}, 
                        'joint_2': {}, 
                        'Bochner_grid': {'num_features': 10000},
                        'Bochner_opt': {'num_features': 10000,
                                        'num_restarts': 10}
                       }
              }

kw_run = {'path_data': './data/data_2d_0/',
          'save_meta': True, 
          'path_models': './models/model_space_2d_0/' 
          }

kw_plot_box = {'save': True, 
               'benchmark': 'joint_1', 
               'bins': 20
               }

kw_plot_samples = {'labels': [['joint_1', 'joint_2'], 
                              ['joint_1', 'Bochner_grid'],
                              ['joint_1', 'Bochner_opt'], 
                              ['joint_1', 'SGD_EI'], 
                              ['joint_1', 'SGD_LCB']], 
                   'save': True, 
                   'bins': 20
                   }

kw_plot_traj = {'save': True, 
                's_names': ['SGD_EI', 'SGD_LCB']
                }

kws = {'kw_prior': kw_prior, 
       'kw_prior_plot': kw_prior_plot, 
       'kw_samplers': kw_samplers, 
       'kw_run': kw_run, 
       'kw_plot_box': kw_plot_box, 
       'kw_plot_samples': kw_plot_samples, 
       'kw_plot_traj': kw_plot_traj
       }

file_to_save = open(path_to_save + 'params_0', 'wb')
pickle.dump(kws, file_to_save)
file_to_save.close()



