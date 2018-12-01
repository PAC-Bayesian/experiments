import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import GPy
import time
from .prepare import Prior, Post, AdditivePrior, AdditivePost
from .utils import transform_grid_X, transform_grid_Y, offset_woodbury
from functools import reduce


class TSGrid(object):
    """Construct a TS sampler on grid."""
    def __init__(self, prior, num_samples=None, name=None, target='f'):
        """
        : param prior: Prior 
                prior GP to sample from
        : num_samples: NoneType or int
                number of samples to draw
        : name: NoneType or str
                name of the sampler
        : target: str
                target of GP value to draw,
                the default is noiseless function (target='f')
        """
        self.post = prior
        self.is_post = isinstance(prior, Post)
        self.ind_new = prior.model.X.shape[0] if self.is_post else 0
        self.model_init = prior.model
        self.input_dim = prior.input_dim
        if self.input_dim < 4: 
            self.x_grid = transform_grid_X(prior.x_grid)
            if self.is_post:
                self.x_grid = np.vstack([prior.model.X, self.x_grid])
            self.num_grids = self.x_grid.shape[0]
        self.bounds = prior.bounds
        self.num_samples = num_samples
        self.name = name
        self.target = target

    def draw_samples(self):
        """Draw num_samples of values on x_grid from prior."""
        pass
    
    def get_minimum(self):
        """Return f_min."""
        pass

    def get_minimizer(self):
        """Return x_argmin."""
        pass    

    def summarize_samples(self):
        pass

    def __repr__(self):
        return 'TSGrid: {} with {} samples'.format(self.name, self.num_samples)


class TSGridJoint(TSGrid):
    """ Construct a TS sampler jointly drawing on grid."""

    def __init__(self, prior, num_samples=100, name='joint', target='f', **kw):
        super(TSGridJoint, self).__init__(prior=prior, num_samples=num_samples, name=name, target=target)
        self.is_additive_post = isinstance(self.post, AdditivePost)

    def drawing(self, model, x_grid, full_cov=True):
        """General drawling funciton given model and x_grid."""
        if self.target == 'Y':
            return model.posterior_samples(x_grid, full_cov=full_cov, size=self.num_samples).T
        else:
            return model.posterior_samples_f(x_grid, full_cov=full_cov, size=self.num_samples).T

    def draw_samples(self, seed=None, verbose=False):
        """Draw samples."""
        if seed is not None:
            np.random.seed(seed)

        if self.is_additive_post:               
            samples = reduce(np.add, [self.drawing(p.m_prior, self.x_grid[:, self.post.active_dims[i]]) \
            for i, p in enumerate(self.post.posts)])
        elif self.is_post:
            samples = self.drawing(model=self.post.m_prior, x_grid=self.x_grid) 
        else:
            samples = self.drawing(model=self.model_init, x_grid=self.x_grid) 
            
        self.value_samples = samples
        
    def get_minimum(self):
        return self.value_samples.min(axis=1)[:, np.newaxis]

    def get_minimizer(self):
        return self.x_grid[self.value_samples.argmin(axis=1)]

    def summarize_samples(self):
        self.value_min = self.get_minimum()
        self.x_argmin = self.get_minimizer()



class TSGridSeq(TSGridJoint):
    """Construct a TS sampler sequentially on grid."""
    
    def __init__(self, prior, num_samples=100, name='seq', target='f', **kw):
        """
        : param order: str
                sampling order of grid points 
        : num_anim: int
                number of animations for sample path
        """
        order = kw.get('order', 'asc')
        num_anim = kw.get('num_anim', 3)
        super(TSGridSeq, self).__init__(prior=prior, num_samples=num_samples, name=name, target=target)
        self.order = order
        self.num_anim = num_anim
        self.log_skip = np.int(self.num_samples ** 0.5)
        self.anim_skip = np.floor_divide(self.num_samples, self.num_anim)

    def draw_samples(self, seed=None, cond=True, verbose=True):
        """
        Main sampler to draw sample paths and log all needed data.

        : param cond: bool
            whether to calculate and record the condition number of woodbury matrix 
            after updating at each grid points
        """
        if seed is not None:
            np.random.seed(seed)

        self.value_samples = np.empty((self.num_samples, self.num_grids))
        self._set_ind_grid()
        self._set_cond_woodbury(cond)
        self._set_model_space()

        for n in range(self.num_samples):
            self._order_ind_grid()
            self.model_seq = GPy.models.GPRegression.from_gp(self.model_init) #deep copy the initial GP model
            self._store_model_seq(n)

            start_time = time.clock() #start clock timing
            for i in self.ind_grid:               
                x_new = self.x_grid[i][np.newaxis, :] #current grid point to update               
                value_new = self._sim_target_value(x_new)
                self.value_samples[n, i] = value_new
                self._record_cond_woodbury(cond, n=n, i=i)              
                self._update_model_seq(x_new, value_new)
                self._store_model_seq(n)
        
            if verbose:
                self._show_duration(start_time, n)

    def _set_model_space(self):
        """
        Initialize space to store all models of sample paths during sequential sampling.
        """
        self.model_space = [[] for s in range(self.num_anim + 1)]

    def _store_model_seq(self, n):
        if n % self.anim_skip == 0:
            i = n // self.anim_skip
            self.model_space[i].append(GPy.models.GPRegression.from_gp(self.model_seq))

    def _sim_target_value(self, x_new):
        if self.target == 'Y':
            value_new = self.model_seq.posterior_samples(x_new, full_cov=True, size=1) 
        else:
            if not self.is_post:
                # self._offset_woodbury()
                offset_woodbury(self.model_seq, self.model_init.num_data)
            value_new = self.model_seq.posterior_samples_f(x_new, full_cov=True, size=1)
        return value_new

    # def _offset_woodbury(self):  
    #     """
    #     Offset the automatically added noise variance to the Woodbury matrix by GPy
    #     if the target is the latent function value f.
    #     """     
    #     ind_diag = (np.arange(self.model_init.num_data, self.model_seq.X.shape[0]), ) * 2
    #     var = self.model_seq.likelihood.gaussian_variance().values
    #     self.model_seq.posterior._K[ind_diag] -= var
    #     self.model_seq.posterior._K[np.diag_indices(self.model_seq.X.shape[0])] -= (1 - 1e-2) * 1e-8

    #     post, *_ = self.model_seq.inference_method.inference(kern=self.model_seq.kern, \
    #     X=self.model_seq.X, Y=self.model_seq.Y, \
    #     likelihood=self.model_seq.likelihood, \
    #     K=self.model_seq.posterior._K)
    #     self.model_seq.posterior = post
         
    def _show_duration(self, start_time, n):
        """
        Display the drawing process.
        """
        if n % self.log_skip == 0:
            print('Sample Path {0}:'.format(n), end=' ')
            print('{0} seconds'.format(time.clock() - start_time))

    def _set_ind_grid(self):
        """
        Set the sampling order of grid points. 
        """
        ind_grid = np.arange(self.num_grids)
        if self.order == 'asc':
            self.ind_grid = ind_grid
        elif self.order == 'dsc':
            self.ind_grid = ind_grid[::-1]
        else:
            self.ind_grid = np.random.permutation(ind_grid)
        
    def _order_ind_grid(self):
        """
        Determine grids index order for sequential sampling:
        if in 'shuf' order, shuffle ind_grid, otherwise stay the same as initialized
        """
        if self.order == 'shuf':
            np.random.shuffle(self.ind_grid)

    def _update_model_seq(self, x, y):
        """
        Add new sampled value and location to the model.
        """
        X_all = np.concatenate((self.model_seq.X, x)) #combine data
        Y_all = np.concatenate((self.model_seq.Y, y))
        self.model_seq.set_XY(X_all, Y_all) #posterior inference

    def _set_cond_woodbury(self, cond):
        if cond:
            self.cond_woodbury = np.zeros_like(self.value_samples)

    def _record_cond_woodbury(self, cond=True, *, n, i):
        if cond:
            self.cond_woodbury[n, i] = self.get_cond_woodbury()

    def get_cond_woodbury(self):
        return np.linalg.cond(self.model_seq.posterior.woodbury_inv)

    def summarize_samples(self):
        super(TSGridSeq, self).summarize_samples()
        try:
            self.cond_woodbury
        except AttributeError:
            pass
        else:
            self.mean_cond_woodbury = self.cond_woodbury.mean(axis=0)
        
    def __repr__(self):
        return super(TSGridSeq, self).__repr__() + ' in {} order'.format(self.order)