import numpy as np
import GPy
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from ..gp_ts.prepare import Prior, AdditivePrior
from .plot_helpers import contour_helper

class PriorPlot(object):
    def __init__(self, prior, figsize=None, save=False, path='./'):
        self.prior = prior
        if isinstance(prior, AdditivePrior):
            self.ppc = [PriorPlot(prior_comp, figsize, save, path) for prior_comp in prior.priors]
        self.figsize = figsize
        self.save = save
        self.path = path
    
    def plot_f_true(self, title='True func f', file_name='f_true', ext='.pdf', **kw):
        figsize = kw.get('figsize', self.figsize)
        
        try:
            x_grid = self.prior.get_grid()
        except AssertionError:
            x_grid = self.prior.gen_grid()

        if not x_grid or self.prior.input_dim > 2:
            print("Cannot plot high-dimenional function")
            return None

        plt.figure(figsize=figsize)
        if self.prior.input_dim == 1:
            plt.plot(*x_grid, self.prior.f_true(*x_grid))
        elif self.prior.input_dim == 2:
            plt.contourf(*x_grid, self.prior.f_true(*x_grid))
            plt.clabel(plt.contour(*x_grid, self.prior.f_true(*x_grid), colors='black'))
        if title:
            plt.title(title)
        plt.tight_layout()
        if self.save:
            plt.savefig(self.path + file_name + ext)
        plt.clf()
        plt.close()
    
    def plot_f_true_comp(self, title="True func f comp", file_name = "add_f_true", ext='.pdf', **kw):
        figsize = kw.get('figsize', self.figsize)
        
        try: 
            hasattr(self, 'ppc')
        except:
            self.plot_f_true()
        else:
            for i in range(len(self.ppc)):
                if not title:
                    title_comp = None
                else:
                    title_comp = title + '{}'.format(i)
                file_name_comp = file_name + "_comp_{}".format(i)
                self.ppc[i].plot_f_true(title=title_comp, file_name=file_name_comp, ext=ext, figsize=figsize, **kw)

    def plot_model(self, title='Fitted GP model as prior', file_name='fitted_GP_prior', ext='.pdf', with_latent=True, **kw):
        assert hasattr(self.prior, 'model'), "need to fit a GP model first, call 'fit_GP_model' on prior"
        figsize = kw.get('figsize', self.figsize)
        self.prior.model.plot(plot_limits=tuple(zip(*self.prior.bounds)), figsize=figsize)
        if title:
            plt.title(title)
        if self.save:
            plt.savefig(self.path + file_name + ext)
        if with_latent:
            self.prior.model.plot_latent(plot_limits=tuple(zip(*self.prior.bounds)), figsize=figsize)
            if title:
                plt.title(title + "(latent)")
            if self.save:
                plt.savefig(self.path + "latent_" + file_name + ext)
        plt.clf()
        plt.close()

    def plot_model_comp(self, title='Fitted GP model as prior comp', file_name='add_fitted_GP_prior', ext='.pdf', with_latent=True, **kw):
        try: 
            hasattr(self, 'ppc')
        except:
            self.plot_model()
        else: 
            for i in range(len(self.ppc)):
                if not title:
                    title_comp = None
                else:
                    title_comp = title + '{}'.format(i)
                file_name_comp = file_name + "_comp_{}".format(i) + ext
                self.ppc[i].plot_model(title_comp, file_name_comp, ext, with_latent)
        
    def plot_contour_2d(self, lower=2.5, upper=97.5, levels=10, scatter=False, \
    file_name='fitted_GP_prior_contour', ext='.pdf', latent=True, **kw):
        assert hasattr(self.prior, 'model'), "need to fit a GP model first, call 'fit_GP_model' on prior"
        assert self.prior.input_dim == 2, "only plot 2d GP model"
        figsize = kw.get('figsize', self.figsize)
        plot_limits = kw.get('plot_limits', tuple(zip(*self.prior.bounds)))
        contour_helper(self.prior.model, figsize=figsize, plot_limits=plot_limits, 
        latent=latent, lower=lower, upper=upper, levels=levels, scatter=scatter)
        if self.save:
            if latent:
                file_name = 'latent_' + file_name
            plt.savefig(self.path + file_name + ext)
        plt.clf()
        plt.close()