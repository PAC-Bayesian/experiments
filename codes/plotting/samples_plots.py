import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.colors import LogNorm
import seaborn as sns
import pickle



class SamplesPlots(object):
    """Contrast plots of two samplers"""

    def __init__(self, samplers=None, label=None, figsize=plt.rcParams['figure.figsize'], save=False, path_to='./', 
    path_from='./data/'):
        self.samplers = samplers
        self.figsize = figsize
        self.save = save
        self.path_to = path_to
        self.label = label
        self.construct_samples(path_from)

    def construct_samples(self, path_from=None):
        self.samples_x_argmin = []
        self.samples_value_min = []
        self.samples_bounds = []
        self.samples_input_dim = []
        if not self.samplers:
            assert self.label, 'both samplers and label are not provided'
            file_to_load = open(path_from , 'rb')
            data = pickle.load(file_to_load)
            file_to_load.close()
            for name in self.label:
                self.samples_x_argmin.append(data['x_argmin'][name])
                self.samples_value_min.append(data['value_min'][name])
                self.samples_bounds.append(data['meta_data'][name].get('bounds', None))
                self.samples_input_dim.append(data['meta_data'][name].get('input_dim', None))
        else:
            for s in self.samplers:
                self.samples_x_argmin.append(s.x_argmin)
                self.samples_value_min.append(s.value_min)
                self.samples_bounds.append(s.bounds)
                self.samples_input_dim.append(s.input_dim)

    def contrast_hist(self, bins=20, file_name='1d-hist-', ext='.pdf', label=None, **kw):
        """Plot 1d-hist densities of argmin and minimum of two samplers for contrast."""
        set_title = kw.get("set_title", False)
        figsize = kw.get('figsize', self.figsize)
        bounds = kw.get('bounds', self.samples_bounds[0])
        if len(bounds) == 1:
            bounds = bounds[0]

        x_argmin_data = np.hstack(self.samples_x_argmin)
        value_min_data = np.hstack(self.samples_value_min)

        if not label:
            label = self.label
        assert type(label) == list and len(label) == 2
        file_name += '-'.join(label)

        plt.figure(figsize=figsize)
        plt.hist(x_argmin_data, bins=bins, density=True, \
        label = label, range=bounds, **kw)
        plt.legend()
        plt.xlabel('x')
        if set_title:
            plt.gca().set_title('x_argmin hist')
        plt.tight_layout()
        if self.save:
            plt.savefig(self.path_to + file_name + '-x_argmin' + ext)
        plt.clf()
        plt.close()

        ## KDE density x_argmin
        plt.figure(figsize=figsize)
        sns.distplot(x_argmin_data[:, 0], kde=True, hist=False, rug=True, kde_kws={'clip': bounds}, label=label[0])
        sns.distplot(x_argmin_data[:, 1], kde=True, hist=False, rug=True, kde_kws={'clip': bounds}, label=label[1])
        plt.xlim(bounds)
        plt.legend()
        plt.xlabel('x')
        if set_title:
            plt.gca().set_title('x_argmin KDE')
        plt.tight_layout()
        if self.save:
            plt.savefig(self.path_to + 'KDE-' + file_name + '-x_argmin' + ext)
        plt.clf()
        plt.close()

        plt.figure(figsize=figsize)
        plt.hist(value_min_data, bins=bins, density=True, \
        label = label, **kw)
        plt.legend()   
        plt.xlabel('f')
        if set_title:
            plt.gca().set_title('value_min hist')
        plt.tight_layout()
        if self.save:
            plt.savefig(self.path_to + file_name + '-value_min' + ext)
        plt.clf()
        plt.close()

        ## KDE density value_min
        plt.figure(figsize=figsize)
        sns.distplot(value_min_data[:, 0], kde=True, hist=False, rug=True, label=label[0])
        sns.distplot(value_min_data[:, 1], kde=True, hist=False, rug=True, label=label[1])
        plt.legend()
        plt.xlabel('f')
        if set_title:
            plt.gca().set_title('value_min KDE')
        plt.tight_layout()
        if self.save:
            plt.savefig(self.path_to + 'KDE-' + file_name + '-value_min' + ext)
        plt.clf()
        plt.close()

    def contrast_hist_2d(self, bins=25, file_name='2d-hist-', ext='.pdf', label=None, **kw):
        """Plot 2d-hist densities of argmin of two samplers for contrast."""
        figsize = kw.get('figsize', self.figsize)


        assert type(label) == list and len(label) == 2
        file_name += '-'.join(label)

        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(121)
        hist1 = ax1.hist2d(self.samples_x_argmin[0][:,0], self.samples_x_argmin[0][:,-1], normed=True,\
        bins=20, range=self.samples_bounds[0], norm=LogNorm())
        fig.colorbar(hist1[-1])
        ax1.set_title(label[0])
        
        ax2 = fig.add_subplot(122)
        hist2 = ax2.hist2d(self.samples_x_argmin[1][:,0], self.samples_x_argmin[1][:,-1], normed=True, \
        bins=20, range=self.samples_bounds[1], norm=LogNorm())
        fig.colorbar(hist2[-1])
        ax2.set_title(label[1])

        plt.tight_layout()
        if self.save:
            plt.savefig(self.path_to + file_name + '-x_argmin' + ext)

    def contrast_hist_2d_margin(self, bins=25, file_name='margin-2d-hist-', ext='.pdf', label=None, **kw):
        """Plot hist of marginal densities of argmin and minimum of two samplers for contrast."""
        set_title = kw.get("set_title", False)
        figsize = kw.get('figsize', self.figsize)
        bounds = kw.get('bounds', self.samples_bounds[0])
        if len(bounds) == 1:
            bounds = bounds[0]

        x_argmin_data_1 = np.hstack([np.take(self.samples_x_argmin[0], [0], -1), np.take(self.samples_x_argmin[1], [0], -1)])
        x_argmin_data_2 = np.hstack([np.take(self.samples_x_argmin[0], [1], -1), np.take(self.samples_x_argmin[1], [1], -1)])
        value_min_data = np.hstack([self.samples_value_min[0], self.samples_value_min[1]])

        if not label:
            label = self.label
        assert type(label) == list and len(label) == 2
        file_name += '-'.join(label)

        plt.figure(figsize=figsize)
        plt.hist(x_argmin_data_1, bins=bins, density=True, \
        label = label, range=bounds[0], **kw)
        if set_title:
            plt.gca().set_title('x_argmin_1 hist')
        plt.legend()
        plt.tight_layout()
        if self.save:
            plt.savefig(self.path_to + file_name + '-x_argmin_1' + ext)
        
        plt.figure(figsize=figsize)
        plt.hist(x_argmin_data_2, bins=bins, density=True, \
        label = label, range=bounds[1], **kw)
        if set_title:
            plt.gca().set_title('x_argmin_2 hist')
        plt.legend()
        plt.tight_layout()
        if self.save:
            plt.savefig(self.path_to + file_name + '-x_argmin_2' + ext)
        
        plt.figure(figsize=figsize)
        plt.hist(value_min_data, bins=bins, density=True, \
        label = label)
        if set_title:
            plt.gca().set_title('value_min hist')
        plt.legend()
        plt.tight_layout()
        if self.save:
            plt.savefig(self.path_to + file_name + '-value_min' + ext)
