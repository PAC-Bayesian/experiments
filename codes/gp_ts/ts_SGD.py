import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import scipy as sp
from scipy.optimize import minimize_scalar
import GPy
import time
import math
from .ts_grid import TSGrid, TSGridJoint, TSGridSeq
from collections import defaultdict
from .prepare import Prior, Post, AdditivePrior, AdditivePost
from .utils import offset_woodbury


class TSSGD(TSGridSeq):
    """Construct a TS sampler in SGD style."""
    def __init__(self, prior, num_samples=100, name='seq', target='f', **kw):
        """
        : param N_new: int
                number of ramdom samples to start (for exploration) 
        : param iter_max: int
                maximium number of iteration per sampled path
        : param tol: float
                error tolerance for stopping rule
        : param alpha: float
                maximum learning rate in terms of maximum grid_gap of all dimension
        : param beta: float:
                a multipler to indicate minimum learning rate, a fraciton of alpha
        : param search_dir: str
                'mean_GD': search direction is the negative mean of gradient
                'realized_GD': search direction is the negative realized gradient
        : param search_step: str
                'Wolfe': using Wolfe probability to determine step size
                'EI': using EI to determine step size
                'LCB': using LCB to determine step size
                'minimum': the smallest step size alpha * beta
        : param hyper_parameters: tuple
                threshold values (c_1, c_2, c_W) for probabilistic Wolfe conditions
        : param LCB_kappa: str
                "const": constant exploration parameter 1.96 for LCB
                "adapt": adapted exploration parameter for LCB
        : param strong_wolfe: bool
                whether to use Strong Wolfe condition or not
        : param num_candidates: int
                number of candidates per each step search
        """
        order = kw.get('order', 'SGD')
        num_anim = kw.get('num_anim', 3)
        super(TSSGD, self).__init__(prior=prior, num_samples=num_samples, name=name, target=target, order=order, num_anim=num_anim)
        
        self.iter_max = kw.get('iter_max', 100)
        # self.input_dim = prior.input_dim
        self.bounds = np.array(prior.bounds)
        self.tol = kw.get('tol', 0.05)
        self.N_new = kw.get('N_new', 5)
        self.alpha = kw.get('alpha', 0.01) 
        self.beta = kw.get('beta', 0.1)
        self.search_dir = kw.get('search_dir', 'realized_GD')
        self.search_step = kw.get('search_step', 'EI')
        self.hyper_parameters = kw.get('hyper_parameters', (0.05, 0.5, 0.3))
        self.num_candidates = 50
        self.strong_wolfe = kw.get('strong_wolfe', True)
        self.details = kw.get('details', [self.search_step])
        if self.search_step != 'mixed':
            assert self.search_step in self.details, "not include the search_step utility"
        else:
            assert set(self.details) == {'EI', 'LCB'}
        self.min_dist = kw.get('min_dist', 1e-6)
        self.kappa = kw.get('kappa', 1.96)
        self.wolfe_nan = []
        self.min_util = kw.get('min_util', 1e-4)
        self.phi =kw.get('phi', 0.05)
    
    def _reset_model(self, n):
        if self.is_post:
            Y_init = self.post._gen_samples_init_Y(self.model_init.X)
            self.model_init.set_Y(Y_init)
            self.value_samples[n, :self.ind_new] = Y_init.T
            
    def draw_samples(self, seed=None, cond=False, verbose=True, grad=True, random_exp=False):
        """
        Main sampler to draw sample paths and log all needed data.
        
        : param grad: bool
                whether to log gradient information during the sampling
        : param cond: bool
                whether to log conditional number of Woodbury matrix
        """
        if seed is not None:
            np.random.seed(seed)    
       
        self.value_samples = np.full((self.num_samples, self.iter_max), np.nan)
        self.step_logs = np.full_like(self.value_samples, np.nan)
        self.x_samples = np.full((self.num_samples, self.iter_max, self.input_dim), np.nan)
        self._set_gradient(grad)
        self._set_cond_woodbury(cond)
        self._set_model_space()
        self._set_details()

        for n in range(self.num_samples):   
            self._reset_model(n)        
            self.model_seq = GPy.models.GPRegression.from_gp(self.model_init) #deep copy the initial GP model
            self._store_model_seq(n)

            X_new = self._initialize()
            values_new = self._sim_target_value(X_new)
            self.value_samples[n, self.ind_new:self.ind_new+1] = np.min(values_new)
            x_0 = X_new[np.argmin(values_new), np.newaxis]
            self.x_samples[n, self.ind_new:self.ind_new+1] = x_0
            self._update_model_seq(x_0, values_new.min(keepdims=True))
            self._store_model_seq(n)
            counter_bound = 0
            bound_signal = False

            start_time = time.clock() #start clock timing
            for i in range(self.iter_max - 1 - self.ind_new):
                grad_stats = self._predict_gradient(x_0)
                sgd = self._draw_gradient(grad_stats)
                end = self._check_gradient(x_0, sgd, grad_stats)  
                
                if grad:
                    self.gradient[n, i+self.ind_new] = sgd
                    self.mean_gradient[n, i+self.ind_new] = grad_stats[0][0, :, 0]
                
                if end:  
                    if i > math.ceil(math.sqrt(self.input_dim)) and not bound_signal:               
                        break
                self._record_cond_woodbury(cond, n=n, i=i+self.ind_new)     

                x_1, t, bound_signal = self._set_next_location(x_0, sgd, grad_stats, n, i)
               
                end_2 = self._check_location(x_1)
                if end_2:
                    if counter_bound > self.input_dim:
                        break
                    else:
                        counter_bound += 1
                        if random_exp:
                            x_1, t, bound_signal = self._random_rev_next_loc(x_0, sgd)
                        else:
                            self._pop_details(n)
                            x_1, t, bound_signal = self._set_next_location(x_0, -sgd, grad_stats, n, i)
                        
                self.step_logs[n, i+self.ind_new] = t
                self.x_samples[n, i+1+self.ind_new] = x_1
                value_1 = self._sim_without_offset(x_1)
                self.value_samples[n, i+1+self.ind_new] = value_1 

                self._update_model_seq(x_1, value_1)
                self._store_model_seq(n)    
                x_0 = x_1
            
            if verbose:
                self._show_duration(start_time, n)

    def _random_rev_next_loc(self, x_0, s=None):
        b = self.bounds + np.array([self.min_dist, -self.min_dist])
        x_1 = np.random.uniform(*zip(*b), size=(1, self.input_dim))
        t = 0
        bound_signal = False
        return x_1, t, bound_signal

    def _set_details(self):
        if self.details:
            self.cands_info = [[] for n in range(self.num_samples)]
            self.utilities_info = {}
            for k in self.details:
                self.utilities_info[k] = [[] for n in range(self.num_samples)]

    def _initialize(self):
        """
        Draw N_new locations randomly in uniform within bounds to initialize.
        """
        X_new = []
        for b in self.bounds:
            X_new.append(np.random.uniform(*b, size=(self.N_new, 1)))
        return np.hstack(X_new)

    def _sim_without_offset(self, x_new):
        """
        Simulate a new observation or latent value without offset Woodbury matrix.
        """
        if self.target == 'Y':
            value_new = self.model_seq.posterior_samples(x_new, full_cov=True, size=1) 
        else:
            value_new = self.model_seq.posterior_samples_f(x_new, full_cov=True, size=1)
        return value_new

    def _set_gradient(self, grad):
        """
        Initialize arrays for gradient information logs.
        """
        if grad:
            self.gradient = np.full_like(self.x_samples, np.nan)
            self.mean_gradient = np.full_like(self.x_samples, np.nan)
    
    def _predict_gradient(self, x):
        """
        Predict gradient mean and covariance at location x.
        """
        if self.target == "f":
            if not self.is_post:
                # self._offset_woodbury()
                offset_woodbury(self.model_seq, self.model_init.num_data)
        mu_grad, Cov_grad = self.model_seq.predict_jacobian(x, full_cov=True)
        if self.is_post:
            mu_grad += self.post.mean_func.gradients_X(np.ones([x.shape[0], 1]), x)[:, :, np.newaxis]
        return mu_grad, Cov_grad
    
    def _draw_gradient(self, grad_stats): 
        """
        Draw a realized gradient with the provided mean and covariance.
        """
        if self.search_dir == "random":
            gap = self.bounds[:, -1] - self.bounds[:, 0]
            gradient = np.random.uniform(low=-gap, high=gap, size=(1, self.input_dim))   
        else:
            mu, Cov = grad_stats
            gradient = np.random.multivariate_normal(mu[0, :, 0], Cov[0, 0, :, :], 1)
        return gradient
    
    def _check_gradient(self, x_0, gradient, grad_stats):
        """
        Check whether the norm of mean (posterior) gradient is within the tolerence.
        """

        return np.linalg.norm(grad_stats[0][0, :, 0]) < self.tol 
        
    def _get_step_max(self, x_0, s):
        """
        Calculate the maximum step t for each dimension on a cube domain, then
        return the minimum of these step.
        
        : param x_0: current location
        : param s: current search direction
        """
        t_cands = []
        x_0 = x_0.flatten()
        s = s.flatten()
        for i in range(self.input_dim):
            if s[i] != 0:
                t_1 = (self.bounds[i][0] - x_0[i]) / s[i]
                t_2 = (self.bounds[i][-1] - x_0[i]) / s[i]
                t_cands.append(max(t_1, t_2))
        return min(t_cands)

    def utility_opt(self, x_0, s, i):
        def step_rand_min():
            mid = (lr_min + self.min_dist) / 2
            diff = (np.random.rand() - 0.5) * (lr_min - self.min_dist) * (1 - self.beta)
            return  mid + diff

        def opt_util():
            for k in self.details:
                if k == "EI":
                    if t_max < lr_min:
                        t_cands.append(lr_min)
                        utilities[k].append(self._calculate_EI(x_0=x_0, y_0=y_0, t=lr_min, s=s))
                    else:
                        res = minimize_scalar(EI_wrapper, bounds=(self.min_dist, t_max), method='bounded')
                        t_opt = res.x
                        utilities[k].append(self._calculate_EI(x_0=x_0, y_0=y_0, t=t_opt, s=s))
                        t_select = step_rand_min() if utilities[k][0] < self.min_util else t_opt  
                        t_cands.append(t_select)
                        # t_cands.append(t_max)

                if k == "LCB":
                    if t_max < lr_min:
                        t_cands.append(lr_min)
                        utilities[k].append(self._calculate_LCB(x_0=x_0, t=lr_min, s=s))
                    else:
                        res = minimize_scalar(LCB_wrapper, bounds=(self.min_dist, t_max), method='bounded')
                        t_opt = res.x
                        utilities[k].append(self._calculate_LCB(x_0=x_0, t=t_opt, s=s))
                        t_select = step_rand_min() if abs(utilities[k][0] - y_0) < self.min_util * self.kappa else t_opt  
                        t_cands.append(t_select)
                        # t_cands.append(t_max)
            t_cands.append(t_max)
        
        def EI_wrapper(t):
            return -self._calculate_EI(x_0=x_0, y_0=y_0, t=t, s=s, phi=self.phi)
        
        def LCB_wrapper(t):
            return -self._calculate_LCB(x_0=x_0, t=t, s=s, kappa=self.kappa)

        y_0 = self.model_seq.Y[-1]
        t_max = self._get_step_max(x_0, s)
        lr_min = self.alpha * self.beta
        utilities = defaultdict(list)
        t_cands = []
        
        
        opt_util()
        if self.search_step == 'mixed':
            value_cands = [self._sim_without_offset(x_new=x_0 + s * t) for t in t_cands] 
            ind = np.argmin(value_cands)
        else:
            ind = 0
        return t_cands[ind], utilities, t_cands
    
    def _calculate_LCB(self, x_0, t, s, i=0, kappa=1.96):
        x_t = x_0 + s * t
        mu, Cov = self.model_seq.predict_noiseless(x_t, full_cov=True)
        if np.float(Cov) <= 0:
            print("LCB_Cov might cause NaN")
            Cov = abs(Cov)

        sigma = np.sqrt(Cov)
        # if self.LCB_kappa == 'adapt':
        #     eps = 0.1
        #     kappa = np.sqrt(2 * np.log((i + 1) ** (self.input_dim / 2 + 2) * np.pi ** 2 / (3 * eps)))
        alpha = -mu + kappa * sigma
        return alpha

    def _calculate_EI(self, x_0, y_0, t, s, phi=0.05):
        x_t = x_0 + s * t
        mu, Cov = self.model_seq.predict_noiseless(x_t, full_cov=True)
        if np.float(Cov) <= 0:
            print("EI_Cov might cause NaN")
            Cov = abs(Cov)

        sigma = np.sqrt(Cov)
        gamma = (y_0 - mu + phi) / sigma
        alpha = sigma * (gamma * sp.stats.norm.cdf(gamma) + sp.stats.norm.pdf(gamma))
        return alpha

    def _get_wolfe_prob(self, x_0, t, s):
        """wrapper to calculate wolfe probability at location x_0 + t * s"""
        wolfe = self._get_wolfe_variables(x_0=x_0, t=t, s=s)
        wolfe_normalized = self._normalize_wolfe_variables(*wolfe)
        wolfe_prob = self._calculate_wolfe_prob(*wolfe_normalized)
        if np.isnan(wolfe_prob):
            self.wolfe_nan.append({"x_0":x_0, "t":t, "s":s, "wolfe":wolfe, "wolfe_norm":wolfe_normalized})
            wolfe_prob = 0
        return wolfe_prob

    def _normalize_wolfe_variables(self, m, C, strong):
        """
        Normalize wolfe variables to obtain the integral limits and correlation parameter.
        """
        m_a, m_b = m
        C_aa, C_ab, C_bb = C
        c_2, mu_prime_0, d2_k_d0_d0 = strong

        # avoid negative values that produce NaN when calling np.sqrt()
        C_aa = abs(C_aa)
        C_bb = abs(C_bb)
        d2_k_d0_d0 = abs(d2_k_d0_d0)

        lb = np.array([-m_a / np.sqrt(C_aa), -m_b / np.sqrt(C_bb)]).flatten()
        rho = C_ab / np.sqrt(C_aa * C_bb)
        if self.strong_wolfe: 
            b_bar = 2 * c_2 * (abs(mu_prime_0) + 2 * np.sqrt(d2_k_d0_d0)) 
            ub_strong = (b_bar - m_b) / np.sqrt(C_bb)
            ub = np.array([np.Infinity, ub_strong])
        else:
            ub = np.array([np.Infinity, np.Infinity])
        return [lb, ub], rho

    def _calculate_wolfe_prob(self, bounds_wolfe, rho):
        """
        To calculate the bivariate Gaussian integral.
        """
        lb, ub = bounds_wolfe
        mu = np.zeros(2)
        S = np.eye(2)
        S[0, 1] = S[1, 0] = rho
        # print("S:", S)
        prob_wolfe, _ = sp.stats.mvn.mvnun(lb, ub, mu, S, abseps=1e-12)
        return prob_wolfe

    def _get_wolfe_variables(self, x_0, t, s):
        """
        To collect all Wolfe variables in use.
        """
        x_t = x_0 + s * t
        c_1, c_2 = self.hyper_parameters[:2]
        
        mu, Cov = self._get_mu_cov(x_0, x_t)
        mu_prime, Cov_prime_prime = self._get_mu_cov_prime(x_0, x_t)
        Cov_prime_0_t = self._get_gradients_X(x_0, x_t)
        Cov_prime_t_0 = self._get_gradients_X(x_t, x_0)
        Cov_prime_0_0 = self._get_gradients_X(x_0, x_0)
        Cov_prime_t_t = self._get_gradients_X(x_t, x_t)

        ## get posterior mean
        mu_0 = mu[0]
        mu_t = mu[1]
        mu_prime_0 = s.dot(mu_prime[0, :, 0])
        mu_prime_t = s.dot(mu_prime[1, :, 0])

        ## get posterior variance
        k_00 = np.array([Cov[0, 0]])
        k_0t = np.array([Cov[0, 1]])
        k_tt = np.array([Cov[1, 1]])
        d2_k_d0_d0 = s @ Cov_prime_prime[0, 0, :, :] @ s.T
        d2_k_d0_dt = s @ Cov_prime_prime[0, 1, :, :] @ s.T
        d2_k_dt_dt = s @ Cov_prime_prime[1, 1, :, :] @ s.T

        ## get posterior covariance
        d_k_0_dt = np.dot(Cov_prime_0_t, s.T)
        d_k_t_d0 = np.dot(Cov_prime_t_0, s.T)
        d_k_0_d0 = np.dot(Cov_prime_0_0, s.T)
        d_k_t_dt = np.dot(Cov_prime_t_t, s.T)
    
        ## calculate mean of a_t, b_t
        m_a = mu_0 - mu_t + c_1 * t * mu_prime_0
        m_b = mu_prime_t - c_2 * mu_prime_0

        ## calculate variance and covariance of a_t, b_t
        C_aa = k_00 + (c_1 * t) ** 2 * d2_k_d0_d0 + k_tt + 2 * (c_1 * t * (d_k_0_d0 - d_k_t_d0) - k_0t)
        C_bb = c_2 ** 2 * d2_k_d0_d0 - 2 * c_2 * d2_k_d0_dt + d2_k_dt_dt
        C_ab = -c_2 * (d_k_0_d0 + c_1 * t * d2_k_d0_d0) + c_2 * d_k_t_d0 + d_k_0_dt + \
        c_1 * t * d2_k_d0_dt - d_k_t_dt
        return ([m_a, m_b], [C_aa, C_ab, C_bb], [c_2, mu_prime_0, d2_k_d0_d0])
        
    def _get_mu_cov(self, x_0, x_t):
        """zero prime"""
        mu, Cov = self.model_seq.predict_noiseless(np.vstack([x_0, x_t]), full_cov=True)
        return mu, Cov

    def _get_mu_cov_prime(self, x_0, x_t):
        """double prime"""
        mu_prime, Cov_prime = self.model_seq.predict_jacobian(np.vstack([x_0, x_t]), full_cov=True)
        return mu_prime, Cov_prime

    def _get_gradients_X(self, x_0, x_t):
        """single prime"""
        d_k_0_dt = self.model_seq.kern.gradients_X(np.eye(1), x_t, x_0) 
        d_k_tilde_0_dt = d_k_0_dt + \
        self.model_seq.kern.K(x_0, self.model_seq.X) @ \
        self.model_seq.posterior.woodbury_inv @ \
        self.model_seq.kern.gradients_X(np.eye(1), self.model_seq.X, x_t)
        return d_k_tilde_0_dt
    
    def _record_details(self, utilities, t_cands, n):
        if self.details:
            self.cands_info[n].append(t_cands)
            for k in self.details:
                self.utilities_info[k][n].append(utilities[k])

    def _pop_details(self, n):
        if self.details:
            del self.cands_info[n][-1]
            for k in self.details:
                del self.utilities_info[k][n][-1]

    def _set_next_location(self, x_0, gradient, grad_stats, n, i):
        """
        New location policy in SGD style with step size of either line_search or minimum learning rate.
        """
        if self.search_dir == "mean_GD":
            s = -grad_stats[0][0, :, 0]
        elif self.search_dir == "realized_GD":
            s = -gradient
        else:
            s = gradient

        if self.search_step != "minimum":
            s = s / np.linalg.norm(s)
            t, utilities, t_cands = self.utility_opt(x_0, s, i)
            self._record_details(utilities, t_cands, n)
            x_1 = x_0 + s * t
            bound_signal = True if abs(t - t_cands[-1]) < self.min_dist else False
        else:
            x_1 = x_0 + s * self.alpha * self.beta
            t = self.alpha * self.beta / np.linalg.norm(s)
            bound_signal = False
        
        return x_1, t, bound_signal

    def _check_location(self, x):
        """
        Check whether the location x is out of bounds.
        """
        return not (np.all(self.bounds[:, 0] <= x) and np.all(x <= self.bounds[:, -1]))

    def get_minimum(self):
        return np.nanmin(self.value_samples, axis=1)[:, np.newaxis]
    
    def get_minimizer(self):
        ind = np.nanargmin(self.value_samples, axis=1)
        return self.x_samples[np.arange(self.num_samples), ind] 

    def summarize_samples(self):
        super().summarize_samples()
        self.num_evals = np.count_nonzero(~np.isnan(self.value_samples), axis=1)

    def __repr__(self):
        rs = super(TSSGD, self).__repr__() + \
        ' (start with {0}, tol={1}, search_step={2}, search_dir={3})'.format(self.N_new, self.tol, self.search_step, self.search_dir)
        return rs.replace('TSGrid', 'TS')


    