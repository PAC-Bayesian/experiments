import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import GPy
from functools import reduce
from operator import add
from .funcs import AdditiveFun
from .utils import transform_grid_X, transform_grid_Y

    
class Prior(object):
    """Construct a prior for samplers."""
    
    def __init__(self, f_true, bounds, input_dim=1, seed=111, **kw):
        """
        : param f_true: function
                true function used
        : param bounds: tuple
                    bounds[i]: tuple
                range of each input dimension
        : param input_dim: int
                dimension of input
        : param seed: int
                random seed for data generation
        """
        assert len(bounds) == input_dim, "inconsistent input dimension"
        self.f_true = f_true
        self.bounds = bounds
        self.input_dim = input_dim
        self.seed = seed
        self.noise_f() # start with an almost noiseless function for frequent use
        self.grid_size = (100,) * self.input_dim # the number of grids of each dimension is 100 by default
                 
    def noise_f(self, noise_std=1e-10):
        """Generate a noisy version of f_true with given noise_std."""
        def f_obs(*args):
            return self.f_true(*args) + noise_std * np.random.randn(*args[0].shape)
        self.f_obs = f_obs

    def _gen_grid(self):	
        """Generate a grid of equal space with grid_size within bounds."""	
        grids = []
        for i in range(self.input_dim):
            grids.append(np.linspace(*self.bounds[i], self.grid_size[i] + 1)[:, np.newaxis])
        return np.meshgrid(*grids, indexing='ij')    
        
    def gen_grid(self, grid_size=None):
        """ 
        Generate a grid of equal space with grid_size within bounds
        and bind it to the Prior object for reference.

        : param grid_size: NoneType or tuple
                    grid_size[i]: int,
                    the default 100 (grid_size=None)
                number of grids of each dimension for reference
        """	
        if grid_size:    
            assert len(grid_size) == self.input_dim, "inconsistent input dimension"
            self.grid_size = grid_size
        self.grid_gap = tuple([(self.bounds[i][-1] - self.bounds[i][0]) / self.grid_size[i] \
        for i in range(self.input_dim)])
        self.x_grid = self._gen_grid()
        
    def get_grid(self):
        """Fetch the x_grid."""
        assert hasattr(self, 'x_grid'), "need to bind a grid first, call 'gen_grid'"
        return self.x_grid

    def _gen_data_init_X(self, N_init, ref=None, on_grids=False, jitter=1, seed=None):
        """
        Generate N_init locations drawn randomly within bounds 
        or around grid points through jitter.

        : param N_init: int
                number of initial data points to fit model
        : param ref: NoneType or 2-d numpy.ndarray 
                specified input locations, 
                the default is a uniformly random draw on each dimension (ref=None)
        : param on_grids: bool
                whether to make initial data inputs close to grid points. 
        : param jitter: float
                deviation to the grid points in proportional to half of grid_gap 
        """	
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
 
        X_ref = []
        for i in range(self.input_dim):
            X_ref.append(np.random.uniform(self.bounds[i][0], self.bounds[i][-1], \
            size=(N_init, 1)))
        if ref is not None:
            assert ref.shape[-1] == self.input_dim, "inconsistent input dimension"
            assert ref.shape[0] <= N_init ** self.input_dim, "inconsistent initial data number"
            N_init -= ref.shape[0]
            for i in range(self.input_dim):
                X_spec = np.take(ref, [i], -1)
                X_ref[i] = np.vstack((X_ref[i][:N_init, :], X_spec))
        X_ref_grid = np.meshgrid(*X_ref, indexing='ij', sparse=True)

        if not on_grids:
            X_init = X_ref_grid
        else:
            assert hasattr(self, 'grid_gap'), "need to bind a grid first, call 'gen_grid'"
            for i in range(self.input_dim):
                X_ref[i] = np.floor_divide(X_ref[i] - self.bounds[i][0], self.grid_gap[i]) * self.grid_gap[i] + \
                jitter * np.random.choice([-1,1], size=(N_init,1)) * self.grid_gap[i] / 2 + \
                self.bounds[i][0]
            X_init = np.meshgrid(*X_ref, indexing='ij', sparse=True)
        return X_init

    def _gen_data_init_Y(self, X_init):
        """Generate observations at locations X_init using f_obs or f_true."""
        try:
            self.f_obs
        except AttributeError:		
            print('generate noiseless observation using f_true')
            Y_init = self.f_true(*X_init)
        else:
            Y_init = self.f_obs(*X_init)
        return Y_init

    def gen_data_init(self, N_init, ref=None, on_grids=False, jitter=1, seed=None):
        """
        Generate N_init observed data using f_obs
        on locations drawn randomly within bounds 
        or around grid points through jitter
        and bind it to the Prior object for model fitting.
        """	

        X_init = self._gen_data_init_X(N_init, ref, on_grids, jitter, seed)
        Y_init = self._gen_data_init_Y(X_init)
        self.X_init, self.Y_init = X_init, Y_init
    
    def get_data_init(self):
        """Fetch X_init, Y_init."""
        assert hasattr(self, 'X_init') and hasattr(self, 'Y_init'), "need to bind initial data first, call 'gen_data_init'"
        return self.X_init, self.Y_init
            
    def fit_GP_model(self, noise_var=None, kern=GPy.kern.RBF, verbose=False, **kw):
        """
        Fit a GP regression model using X_init, Y_init. 
        The number of optimization restarts is 10 by default. 

        : param noise_var: NoneType or float
                Gaussian_noise fitting or fixed
        : param kern: subclass of Kern in GPy.kern.src
                kernel to use
        """	
        X_grid, Y_grid = self.get_data_init()
        X, Y = transform_grid_X(X_grid, mesh=True), transform_grid_Y(Y_grid)
        kernel = kern(input_dim=self.input_dim)
        self.model = GPy.models.GPRegression(X, Y, kernel=kernel)
        if noise_var is not None:
            self.model.Gaussian_noise.constrain_fixed(noise_var, warning=False) # fix Gaussian noise variance
        self.model.optimize_restarts(verbose=verbose, **kw) # optimize hyper parameters
        if verbose:
            print(self.model)
            print()
        
class Post(Prior):
    """Construct a posterior for samplers."""

    def __init__(self, prior, seed=111, active_dims=None, target='f', **kw):
        """
        : param prior: Prior
                the prior used to get posterior
        : param seed: int
                random seed for data generation
        : param active_dims: NoneType or 1-d numpy.ndarray
                indices of active dimensions the kernel is working on,
                the default is the same as that of kernel in prior.model
        : param target: str
                working with function value with noise or not,
                the default is noiseless function (target='f')
        """
        self.target = target
        self.m_prior = prior.model
        self.seed = seed
        self.bounds = prior.bounds
        self.input_dim = prior.input_dim
        if hasattr(prior, 'x_grid'):
            self.x_grid = prior.x_grid
            self.grid_gap = prior.grid_gap
        self.kern = self._gen_kern_post(active_dims=active_dims)
        self.mean_func = self._gen_mean_func_post()
        
    def _gen_kern_post(self, active_dims=None):
        """Generate posterior kernel from m_prior."""
        return GPy.kern.src.kern.PosteriorKernel(kernel=self.m_prior.kern, \
        posterior=self.m_prior.posterior, \
        X_obs=self.m_prior.X,\
        active_dims=active_dims)

    def _gen_mean_func_post(self):
        """Generate posterior mean function from m_prior."""
        return GPy.mappings.Kernel_with_A(input_dim=self.input_dim, \
        output_dim=1, \
        Z=self.m_prior.X, \
        kernel=self.m_prior.kern, \
        A=self.m_prior.posterior.woodbury_vector)

    def _gen_samples_init_X(self, N_init=1, ref=None, on_grids=False, jitter=1, seed=None):
        """Generate the initial location X_init."""
        if seed is None:
            seed = self.seed
        
        X_init = transform_grid_X(self._gen_data_init_X(N_init, ref, on_grids, jitter, seed)) 
        return X_init

    def _gen_samples_init_Y(self, X_init):
        """Sample obervations at locations X_init."""
        np.random.seed()
        if self.target == 'Y':
            Y_init = self.m_prior.posterior_samples(X_init, full_cov=True, size=1)
        else:
            Y_init = self.m_prior.posterior_samples_f(X_init, full_cov=True, size=1)   

        return Y_init 

    def gen_samples_init(self, N_init=1, ref=None, on_grids=False, jitter=1, seed=None):
        X_init = self._gen_samples_init_X(N_init, ref, on_grids, jitter, seed)
        Y_init = self._gen_samples_init_Y(X_init)
        return X_init, Y_init

    def fit_GP_model(self, noise_var=None, kern=None, mean_func=None, verbose=True, \
    X_init=None, N_init=1, **kw):
        """
        Fit a GP regression model using X_init, Y_init. 
        The number of optimization restarts is 10 by default. 

        : param noise_var: NoneType or float
                Gaussian_noise to be fixed
        : param kern: NoneType or subclass of Kern in GPy.kern.src
                kernel to use,
                the default is self.kern (kern=None)
        : param mean_func: NoneType or subclass of Mapping in GPy.mappings
                mean function to use,
                the default is self.mean_func (kern=None)
        : param X_init: NoneType or 2-d numpy.ndarray
                specify the initial locations of obervations to include
        : param N_init: int
                number of initial observation to sample
        """	

        if kern is None:
            kern = self.kern
        if mean_func is None:
            mean_func = self.mean_func
        if noise_var is None:
            if self.target == 'f':
                noise_var = 1e-8
            elif self.target == 'Y':
                noise_var = self.m_prior['.*noise'].values()

        kern.fix()
        mean_func.fix()
        X, Y = self.gen_samples_init(N_init=N_init, ref=X_init)
        self.model = GPy.models.GPRegression(X, Y, \
        noise_var=noise_var, \
        kernel=kern, \
        mean_function=mean_func)
        self.model.Gaussian_noise.constrain_fixed(noise_var, warning=False) # fix Gaussian noise variance
        
        if verbose:
            print(self.model)
            print()

class AdditivePrior(Prior):
    """Construct an additive prior for samplers."""

    def __init__(self, priors, seed=111, **kw):
        """
        : param priors: list or tuple
                priors[i]: Prior
                a sequence of priors as components
        """
        assert isinstance(priors, (list, tuple)) and len(priors) >= 2, "Cannot make additive"
        self.priors = priors
        self.f_additive = self._combine_fun()
        self.f_true = self.f_additive.add_f()
        self.input_dim = self._combine_input_dim()
        assert self.input_dim == self.f_additive.get_input_dim(), \
        "Inconsistent input dimension"
        self.bounds = self._combine_bounds()
        self.grid_size_comp = kw.get('grid_size_comp', 50) # same grid_size for each dim by default
        self.grid_size = self._combine_grid_size(self.grid_size_comp)
        # self.grid_gap = self._combine_grid_gap()
        self.seed = seed
    
    def _combine_grid_size(self, grid_size):
        """Combine grid_size of each component prior."""
        if type(grid_size) is int:
            return (grid_size,) * self.input_dim
        else:
            assert isinstance(grid_size, (tuple, list)) and \
            len(grid_size) == self.f_additive.num_comp, "Inconsistent components number"
            assert all([isinstance(gc, tuple) for gc in grid_size]), "Tuple required for component grid_size"
            return reduce(add, grid_size)
    
    def _combine_grid_gap(self):
        """Combine grid_gap of each component prior if it exists."""
        self.grid_gap = reduce(add, [p.grid_gap for p in self.priors])

    def _combine_fun(self):
        """Combine an AdditiveFun object from f_true of component prior."""
        funs = [p.f_true for p in self.priors]
        return AdditiveFun(funs)

    def _combine_input_dim(self):
        """Add the input_dim of each prior component."""
        input_dim = reduce(add, [p.input_dim for p in self.priors])
        return input_dim

    def _combine_bounds(self):
        """Combine bounds of each component prior."""
        bounds = reduce(add, [p.bounds for p in self.priors])
        return bounds

    def gen_grid_comp(self):
        """Generate grid for each component prior."""
        for c in range(self.f_additive.num_comp):
            if isinstance(self.grid_size_comp, int):
                grid_size_comp = (self.grid_size_comp, ) * self.priors[c].input_dim
            else:
                grid_size_comp = self.grid_size_comp[c]
            self.priors[c].gen_grid(grid_size_comp)
        self._combine_grid_gap()

    def _combine_grid(self):
        """Combine grid of each component prior if the additive prior has input_dim <= 3."""
        if self.input_dim < 4:
            return np.meshgrid(*(p.get_grid() for p in self.priors), indexing='ij')
        else:
            print("Not necessary to combine grids")
            return None
    
    def gen_grid(self):
        """Generate and combine grid."""
        self.gen_grid_comp()
        self.x_grid = self._combine_grid()

    def gen_data_init_comp(self, N_init, ref=None, seed=None):
        """Generate initial data for each component prior."""
        if type(N_init) is int:
            N_init = (N_init, ) * self.f_additive.num_comp
        assert type(N_init) is tuple and len(N_init) == self.f_additive.num_comp, \
        "Inconsistent components number"

        if ref is None:
            ref = (None, ) * self.f_additive.num_comp
        assert type(ref) is tuple and len(ref) == self.f_additive.num_comp, \
        "Inconsistent components number" 

        if seed is None:
            seed = self.seed
        
        for c in range(self.f_additive.num_comp):
            self.priors[c].gen_data_init(N_init=N_init[c], ref=ref[c], \
            seed=seed)
            seed += self.f_additive.num_comp

    def _combine_data_init(self):
        """Combine initial data of each component prior."""
        X_init = reduce(add, [p.X_init for p in self.priors])
        X_init_grid = np.meshgrid(*X_init, indexing='ij')

        Y_init = [transform_grid_Y(p.Y_init) for p in self.priors] 
        Y_init_grid = np.meshgrid(*Y_init, indexing='ij', sparse=True)

        self.X_init = X_init_grid
        self.Y_init = reduce(add, Y_init_grid)  

    def gen_data_init(self, N_init=1, ref=None, seed=None):
        """Generate and combine initial data of each component prior."""
        self.gen_data_init_comp(N_init=N_init, ref=ref, seed=seed)
        self._combine_data_init()

    def fit_GP_model_comp(self, noise_var=None, kern=GPy.kern.RBF, verbose=True, **kw): 
        """Fit a GP regression model for each component prior."""
        
        if noise_var is None:
            noise_var = (None, ) * len(self.priors)
        assert type(noise_var) is tuple and len(noise_var) == self.f_additive.num_comp, \
        "Inconsistent components number" 

        if not isinstance(kern, (tuple, list)):
            kern = (kern, ) * len(self.priors)
        assert type(kern) is tuple and len(kern) == self.f_additive.num_comp, \
        "Inconsistent components number"

        for i in range(len(self.priors)):
            self.priors[i].fit_GP_model(noise_var=noise_var[i], kern=kern[i], \
             verbose=verbose, **kw)
        
    def _combine_kernel(self, kern):
        """Combine kernel of GP model for each component prior."""
        kernels = []
        start = 0
        for i in range(len(self.priors)):
            dim = self.priors[i].input_dim
            end = start + dim 
            k = kern[i](input_dim=dim, active_dims=list(range(start, end)))
            kernels.append(k)
            start = end
        return kernels

    def _combine_param(self):   
        """Combine parameters of GP model for each component prior."""
        kernel_params = []
        noise_params = []
        for p in self.priors:
            param_kern, param_noise = p.model.parameters.copy()
            kernel_params.append(param_kern.param_array)
            noise_params.append(param_noise.param_array)
        noise_add = reduce(add, noise_params)
        param_array = kernel_params + [noise_add]
        return np.concatenate(param_array)

    def fit_GP_model(self, free=False, noise_var=None, kern=GPy.kern.RBF, verbose=True, **kw):
        if not isinstance(kern, (tuple, list)):
            kern = (kern, ) * len(self.priors)
        assert type(kern) is tuple and len(kern) == self.f_additive.num_comp, \
        "Inconsistent components number"

        self.fit_GP_model_comp(noise_var=noise_var, kern=kern, verbose=verbose, **kw)

        kernels = self._combine_kernel(kern)
        add_kernel = reduce(add, kernels)

        X_grid, Y_grid = self.get_data_init()
        X, Y = transform_grid_X(X_grid), transform_grid_Y(Y_grid)
        add_model = GPy.models.GPRegression(X, Y, kernel=add_kernel)
        param_array = self._combine_param()
        if free:
            add_model.Gaussian_noise.constrain_fixed(param_array[-1], warning=False)
            add_model.optimize_restarts(verbose=verbose, **kw)
        else:
            add_model.constrain_fixed(param_array)
        self.model = add_model


class AdditivePost(Post, AdditivePrior):
    """Construct an additive prior for samplers."""

    def __init__(self, priors, seed=111, target='f', **kw):
        AdditivePrior.__init__(self, priors, seed, **kw)

        self.target = target
        self.active_dims = self._gen_active_dims()
        self.posts = self._gen_posts()
        self.kern = self._combine_kernel_post()
        self.mean_func = self._combine_mean_func()
    
    def _gen_active_dims(self):
        """Generate active_dims according to input_dim of each component."""
        active_dims = []
        start = 0
        for i in range(len(self.priors)):
            dim = self.priors[i].input_dim
            end = start + dim
            active_dims.append(np.arange(start, end))
            start = end
        return active_dims

    def _gen_posts(self):
        """Combine posterior of each component."""
        posts = []
        for i in range(len(self.priors)):
            post = Post(self.priors[i], seed=self.seed+i+1, active_dims=self.active_dims[i])
            posts.append(post)
        return posts

    def _combine_kernel_post(self):
        """Create an additive posterior kernel."""
        tmp = self.posts[0].kern + self.posts[1].kern
        for i in range(1, len(self.posts) - 1):
            tmp = tmp + self.posts[i + 1].kern
        return tmp
    
    def _combine_mean_func(self):
        """Create an additive posterior mean_func."""
        tmp = GPy.mappings.Additive_with_active_dims(self.posts[0].mean_func, self.posts[1].mean_func, \
        self.active_dims[0], self.active_dims[1])
        for i in range(1, len(self.posts) - 1):
            active_dims1 = np.hstack(self.active_dims[:i + 1])
            active_dims2 = self.active_dims[i + 1]
            tmp = GPy.mappings.Additive_with_active_dims(tmp, self.posts[i + 1].mean_func, \
        active_dims1, active_dims2)
        return tmp

    def _gen_samples_init_X(self, N_init=1, ref=None, on_grids=False, jitter=1, seed=None):
        X_init = []
        """Generate and combine locations of initial samples."""
        for i in range(len(self.posts)):
            # Using slicing instead
            ref_comp = None if ref is None else ref[:, self.active_dims[i]]
            X = self.posts[i]._gen_samples_init_X(N_init=N_init, ref=ref_comp, on_grids=on_grids, jitter=jitter, seed=seed)
            X_init.append(X)
        return np.hstack(X_init)
    
    def _gen_samples_init_Y(self, X_init):
        """Generate and combine values of initial samples."""
        Y_init = []
        for i in range(len(self.posts)):
            # Using slicing instead 
            Y = self.posts[i]._gen_samples_init_Y(X_init[:, self.active_dims[i]])
            Y_init.append(Y)
        return reduce(np.add, Y_init)
    
    def fit_GP_model(self, noise_var=None, kern=None, mean_func=None, verbose=True, \
    X_init=None, N_init=1, **kw):
        Post.fit_GP_model(self, noise_var, kern, mean_func, verbose, X_init, N_init, **kw)

    


