import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import GPy
import GPyOpt
from GPyOpt.optimization.optimizer import OptLbfgs
import time
from .prepare import Prior, Post, AdditivePrior, AdditivePost
from .ts_grid import TSGrid, TSGridJoint
from ..plotting.prior_plot import contour_helper
import gc
gc.enable()

class ThompsonSamplerRBF():
    """Spectral sampling from GP with RBF kernel."""
    
    def __init__(self, model, num_samples=1000, num_features=500):
        """
        : param model: GPy.models.gp_regression.GPRegression
                GPy regression model with and RBF kernel.
        : param num_samples: int
                number of samples to generate.
        : param num_features: int
                number of features used to approximate the samples.
        """
        # Saving the model elements
        self.model = model
        self.Xsamples = model.X
        self.Ysamples = model.Y
        self.num_data = model.num_data
        self.num_samples = num_samples
        self.num_features = num_features
        
        # Saving the model parameters
        self.lenghtscale = 1/(model.kern.lengthscale**2)
        self.variance = model.kern.variance
        self.noise = model.Gaussian_noise[0]
        self.input_dim = model.X.shape[1]
    
        # We sample the Fourier coefficients and keep them fixed thereafter
        self._compute_random_features()
        self._compute_sample_design_matrix()
        # self.generate_samples_coefficients_posterior(num_samples)

        
    def _compute_random_features(self):
        """Generates the random matrices needed to create each one of the samples."""
        self.W = np.random.normal(0,1,self.num_features*self.input_dim).reshape(self.num_features, self.input_dim) * np.sqrt(self.lenghtscale)
        self.b = 2 * np.pi * np.random.rand(self.num_features, 1)
    
    def compute_design_matrix(self,X):
        """
        Computes the design matrix for a give vector X, 
        given that the Fourier components have been sampled.
        """
        return (np.sqrt(2 * self.variance / self.num_features) * np.cos(np.dot(self.W, X.T) + np.repeat(self.b,X.shape[0],axis=1))).T
        
    def _compute_sample_design_matrix(self):
        self.sample_design_matrix =  self.compute_design_matrix(self.Xsamples)
        
    def generate_samples_coefficients_posterior(self,num_samples=100):
        """
        Samples the coefficients of the specified number of samples. 
        The coeffients are saved to later evaluate the samples at any point in the domain.
        """
        design_matrix = self.sample_design_matrix
        
        # white_noise = np.random.normal(0,1,self.num_features*self.num_samples).reshape(self.num_features, self.num_samples)      
        A = np.dot(design_matrix.T,design_matrix) + self.noise*np.identity(self.num_features) 
        A_inv = np.linalg.inv(A)
        mean = np.dot(np.dot(A_inv,design_matrix.T),self.Ysamples)
        Sigma = self.noise * A_inv
        self.samples_coefficients_posterior = np.random.multivariate_normal(mean.flatten(),Sigma,num_samples).T
        gc.collect()

    def evauate_sample_posterior(self,X,index):
        """
        Evaluate the continuous samples of the posterior.

        : param X: 2-d numpy.ndarray 
                locations in which the samples will be evaluated
        : param index: list
                index[i]: int
                list with the indices of the samples that will be evaluated
        """
        Phi = self.compute_design_matrix(np.atleast_2d(np.array(X)))
        return  np.dot(Phi,self.samples_coefficients_posterior[:,index])

class TSBochnerGrid(TSGridJoint):
    def __init__(self, prior, num_samples=None, name='Bochner_grid', target='f', **kw):
        num_features = kw.get('num_features', 1000)  
        super().__init__(prior, num_samples, name, target)
        self.sampler = ThompsonSamplerRBF(self.model_init, num_samples, num_features)
        
    def draw_samples(self, seed=None, verbose=False):
        if seed is not None:
            np.random.seed(seed)
        
        self.sampler.generate_samples_coefficients_posterior(self.num_samples)
        self.value_samples = self.sampler.evauate_sample_posterior(self.x_grid, range(self.num_samples)).T
        gc.collect()

    def show_GP_sample(self, ind=[0], save=False, **kw):
        sample_f = self.sampler.evauate_sample_posterior(self.x_grid, ind)
        if self.input_dim == 1:
            fig = GPy.plotting.plotting_library().figure()
            self.sampler.model.plot_f(figure=fig, plot_limits=self.bounds[0])
            plt.plot(self.x_grid, sample_f, 'r-', label = 'GP sample_f')
            plt.axvline(x=self.x_argmin[ind[0], :], label='Optimum', color='orange')
            plt.legend()
        elif self.input_dim == 2:
            grid_shape = kw.get('grid_shape', (int(self.num_grids ** 0.5), ) * 2)
            x_grid_1 = self.x_grid.take([0], 1).reshape(grid_shape)
            x_grid_2 = self.x_grid.take([1], 1).reshape(grid_shape)
            value_grid = sample_f.reshape(grid_shape)
            plot_limits = tuple(zip(*self.bounds))
            vmin, vmax, levels = contour_helper(self.model_init, plot_limits=plot_limits)
            plt.clf()
            plt.close()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cf = ax.contourf(x_grid_1, x_grid_2, value_grid, vmin=vmin, vmax=vmax, levels=levels, extend='both')
            plt.clabel(plt.contour(x_grid_1, x_grid_2, value_grid, vmin=vmin, vmax=vmax, levels=levels, extend='both'),
            colors="black")
            plt.colorbar(cf) 
            x_opt = self.x_argmin.take(ind, 0)
            ax.scatter(x_opt[:, 0], x_opt[:, 1], color='orange', marker='x')
        if save:
            plt.savefig(kw.get('path_to') + 'Bochner_path' + '_{}'.format(ind[0]) + '.pdf')
        plt.clf()
        plt.close()

    def __repr__(self):
        return 'TS: {} with {} samples'.format(self.name, self.num_samples) + \
        "(num_features={})".format(self.sampler.num_features)
        
class TSBochnerOpt(TSBochnerGrid):
    def __init__(self, prior, num_samples=None, name='Bochner_opt', target='f', **kw):
        num_features = kw.get('num_features', 1000)  
        num_restarts = kw.get('num_restarts', 1) 
        super().__init__(prior, num_samples, name, target, num_features=num_features)
        self.log_skip = np.int(self.num_samples ** 0.5)
        self.num_restarts = num_restarts

    def optimization(self, index, guess_init=1e-6, optimizer=OptLbfgs):
        sample_GP = lambda x: self.sampler.evauate_sample_posterior(x, [index])
        lbbgs_opt = OptLbfgs(self.bounds)
        result_x, result_fx = lbbgs_opt.optimize(x0 = guess_init, f = sample_GP)
        return result_x, result_fx

    def _initialize(self, N_new=1):
        X_new = []
        for b in self.bounds:
            X_new.append(np.random.uniform(*b, size=(N_new, 1)))
        return np.hstack(X_new)

    def draw_samples(self, seed=None, verbose=False):
        if seed is not None:
            np.random.seed(seed)
        
        self.value_samples = np.full((self.num_samples, self.num_restarts), np.nan) 
        self.x_samples = np.full((self.num_samples, self.num_restarts, self.sampler.input_dim), np.nan) 

        self.sampler.generate_samples_coefficients_posterior(self.num_samples)
        for n in range(self.num_samples): 
            start_time = time.clock()
            for i in range(self.num_restarts):
                guess_init = self._initialize()
                self.x_samples[n, i], self.value_samples[n, i] = self.optimization(n, guess_init)
            
            if verbose:
                self._show_duration(start_time, n)
        gc.collect()

    def _show_duration(self, start_time, n):
        """Display the drawing process."""
        if n % self.log_skip == 0:
            print('Sample Path {0}:'.format(n), end=' ')
            print('{0} seconds'.format(time.clock() - start_time))

    def get_minimum(self):
        return np.nanmin(self.value_samples, axis=1)[:, np.newaxis]
    
    def get_minimizer(self):
        ind = np.nanargmin(self.value_samples, axis=1)
        return self.x_samples[np.arange(self.num_samples), ind]

    def __repr__(self):
        return super(TSBochnerOpt, self).__repr__() + \
        "(num_restarts={})".format(self.num_restarts) 
        
