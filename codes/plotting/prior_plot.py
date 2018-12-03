import numpy as np
import GPy
from GPy.plotting.gpy_plot.plot_util import helper_predict_with_model, helper_for_plot_data, \
get_which_data_ycols, get_x_y_var
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from ..gp_ts.prepare import Prior, AdditivePrior


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

def contour_helper(model, figsize=None, plot_limits=None, lower=2.5, upper=97.5, levels=10, scatter=True, **kw):
    latent = kw.get('latent', True)
    which_data_rows = kw.get('which_data_rows', range(len(model.X)))
    scatter_color = kw.get('scatter_color', 'black')

    ycols = get_which_data_ycols(model, 'all')
    X = get_x_y_var(model)[0]
    helper_data = helper_for_plot_data(model, X, plot_limits=plot_limits, \
                                    visible_dims=None, fixed_inputs=None, resolution=None)
    
    X_matrix = helper_data[2]
    num_grid = int(np.sqrt(X_matrix.shape[0]))
    X_meshgrid = [X_matrix[:, 0].reshape((num_grid, num_grid)), X_matrix[:, 1].reshape((num_grid, num_grid))]
    if latent:
        helper_prediction = helper_predict_with_model(model, X_matrix, True, False,
                                                (lower, upper),
                                                ycols, None)
    else:
        helper_prediction = helper_predict_with_model(model, X_matrix, False, False,
                                                (lower, upper),
                                                ycols, None)
    mu = helper_prediction[0].reshape(num_grid, num_grid)
    lb = helper_prediction[1][0].reshape(num_grid, num_grid)

    vmin = min(np.amin(mu), np.amin(lb))
    vmax = max(np.amax(mu), np.amax(lb))
    levels = np.linspace(vmin, vmax, levels)

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(121)
    cf1 = ax1.contourf(*X_meshgrid, mu, vmin=vmin, vmax=vmax, levels=levels)
    plt.clabel(plt.contour(*X_meshgrid, mu, vmin=vmin, vmax=vmax, levels=levels), colors="black")
    plt.colorbar(cf1)
    if scatter:
        ax1.scatter(model.X.take([0], 1)[which_data_rows], model.X.take([1], 1)[which_data_rows], color=scatter_color, marker='x')
    ax1.set_title('posterior mean')
    ax1.set_xlabel(r'$\mathregular{x_1}$')
    ax1.set_ylabel(r'$\mathregular{x_2}$')

    ax2 = fig.add_subplot(122)
    cf2 = ax2.contourf(*X_meshgrid, lb, vmin=vmin, vmax=vmax, levels=levels)
    plt.clabel(plt.contour(*X_meshgrid, lb, vmin=vmin, vmax=vmax, levels=levels), colors="black")
    plt.colorbar(cf2)
    ax2.set_title('posterior lower CI')
    ax2.set_xlabel(r'$\mathregular{x_1}$')
    ax2.set_ylabel(r'$\mathregular{x_2}$')
    if scatter:
        ax2.scatter(model.X.take([0], 1)[which_data_rows], model.X.take([1], 1)[which_data_rows], color=scatter_color, marker='x')
    plt.tight_layout()
    return vmin, vmax, levels