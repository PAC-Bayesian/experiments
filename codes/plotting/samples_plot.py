import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.switch_backend('agg')
import seaborn as sns
import pickle
import itertools
from .plot_helpers import contrast_hist_1d, contrast_KDE_1d, contrast_hist_2d, contrast_KDE_2d



class SamplesPlot(object):
    """Contrastive plots of two samplers"""

    def __init__(self, samplers=None, label=None, figsize=None, save=False, path_to='./', 
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

    def contrast_hist_margin_1d(self, bins=20, file_name='1d-hist-', ext='.pdf', label=None, **kw):
        set_title = kw.get("set_title", False)
        figsize = kw.get('figsize', self.figsize)
        bounds = kw.get('bounds', self.samples_bounds[0])

        if not label:
            label = self.label
        assert type(label) == list and len(label) == 2, "only contrast two samplers"
        file_name += '-'.join(label)
        assert self.samples_input_dim[0] == self.samples_input_dim[1], "inconsistent input dimensions"
        input_dim = self.samples_input_dim[0]
        assert input_dim == len(bounds), "inconsistent input dimensions"

        for d in range(input_dim):
            x_argmin_data = np.hstack([self.samples_x_argmin[0].take([d], -1), self.samples_x_argmin[1].take([d], -1)])
            xlabel = 'x' if input_dim == 1 else r'$\mathregular{{x_{}}}$'.format(d+1)
            title_words = 'x_argmin' if input_dim == 1 else 'x_argmin_{}'.format(d+1)
            contrast_hist_1d(x_argmin_data, label=label, figsize=figsize, bins=bins, bounds=bounds[d], xlabel=xlabel)
            if set_title:
                plt.gca().set_title(title_words + ' hist')
            plt.tight_layout()
            if self.save:
                plt.savefig(self.path_to + file_name + '-' + title_words + ext)
            plt.clf()
            plt.close()

            contrast_KDE_1d(x_argmin_data, label=label, figsize=figsize, bounds=bounds[d], xlabel=xlabel)
            plt.tight_layout()
            if set_title:
                plt.gca().set_title(title_words + ' KDE')
            if self.save:
                plt.savefig(self.path_to + 'KDE-' + file_name + '-' + title_words + ext)
            plt.clf()
            plt.close()
        
        value_min_data = np.hstack([self.samples_value_min[0], self.samples_value_min[1]])
        xlabel = 'f'
        title_words = 'value_min'
        contrast_hist_1d(value_min_data, label=label, figsize=figsize, bins=bins, xlabel=xlabel)
        if set_title:
            plt.gca().set_title(title_words + ' hist')  
        plt.tight_layout()
        if self.save:
            plt.savefig(self.path_to + file_name + '-' + title_words + ext)
        plt.clf()
        plt.close()

        contrast_KDE_1d(value_min_data, label=label, figsize=figsize, xlabel=xlabel)
        if set_title:
            plt.gca().set_title(title_words + ' KDE')
        plt.tight_layout()   
        if self.save:
            plt.savefig(self.path_to + 'KDE-' + file_name + '-' + title_words + ext)
        plt.clf()
        plt.close()
        
    # def contrast_hist(self, bins=20, file_name='1d-hist-', ext='.pdf', label=None, **kw):
    #     """Plot 1d-hist densities of argmin and minimum of two samplers for contrast."""
    #     set_title = kw.get("set_title", False)
    #     figsize = kw.get('figsize', self.figsize)
    #     bounds = kw.get('bounds', self.samples_bounds[0])
    #     if len(bounds) == 1:
    #         bounds = bounds[0]

    #     x_argmin_data = np.hstack(self.samples_x_argmin)
    #     value_min_data = np.hstack(self.samples_value_min)

    #     if not label:
    #         label = self.label
    #     assert type(label) == list and len(label) == 2
    #     file_name += '-'.join(label)

    #     plt.figure(figsize=figsize)
    #     plt.hist(x_argmin_data, bins=bins, density=True, \
    #     label = label, range=bounds, **kw)
    #     plt.legend()
    #     plt.xlabel('x')
    #     if set_title:
    #         plt.gca().set_title('x_argmin hist')
    #     plt.tight_layout()
    #     if self.save:
    #         plt.savefig(self.path_to + file_name + '-x_argmin' + ext)
    #     plt.clf()
    #     plt.close()

    #     ## KDE density x_argmin
    #     plt.figure(figsize=figsize)
    #     sns.distplot(x_argmin_data[:, 0], kde=True, hist=False, rug=True, kde_kws={'clip': bounds}, label=label[0])
    #     sns.distplot(x_argmin_data[:, 1], kde=True, hist=False, rug=True, kde_kws={'clip': bounds}, label=label[1])
    #     plt.xlim(bounds)
    #     plt.legend()
    #     plt.xlabel('x')
    #     if set_title:
    #         plt.gca().set_title('x_argmin KDE')
    #     plt.tight_layout()
    #     if self.save:
    #         plt.savefig(self.path_to + 'KDE-' + file_name + '-x_argmin' + ext)
    #     plt.clf()
    #     plt.close()

    #     plt.figure(figsize=figsize)
    #     plt.hist(value_min_data, bins=bins, density=True, \
    #     label = label, **kw)
    #     plt.legend()   
    #     plt.xlabel('f')
    #     if set_title:
    #         plt.gca().set_title('value_min hist')
    #     plt.tight_layout()
    #     if self.save:
    #         plt.savefig(self.path_to + file_name + '-value_min' + ext)
    #     plt.clf()
    #     plt.close()

    #     ## KDE density value_min
    #     plt.figure(figsize=figsize)
    #     sns.distplot(value_min_data[:, 0], kde=True, hist=False, rug=True, label=label[0])
    #     sns.distplot(value_min_data[:, 1], kde=True, hist=False, rug=True, label=label[1])
    #     plt.legend()
    #     plt.xlabel('f')
    #     if set_title:
    #         plt.gca().set_title('value_min KDE')
    #     plt.tight_layout()
    #     if self.save:
    #         plt.savefig(self.path_to + 'KDE-' + file_name + '-value_min' + ext)
    #     plt.clf()
    #     plt.close()

    def contrast_hist_margin_2d(self, bins=25, file_name='2d-hist-', ext='.pdf', label=None, **kw):
        """Plot 2d-hist densities of argmin of two samplers for contrast."""
        set_title = kw.get("set_title", False)
        figsize = kw.get('figsize', self.figsize)
        KDE = kw.get('KDE', True)

        if not label:
            label = self.label
        assert type(label) == list and len(label) == 2
        file_name += '-'.join(label)
        assert self.samples_input_dim[0] == self.samples_input_dim[1], "inconsistent input dimensions"
        input_dim = self.samples_input_dim[0]
        ind_pairs = kw.get('ind_pairs', list(itertools.combinations(range(input_dim), 2)))
        
        for ip in ind_pairs:
            bounds = [self.samples_bounds[0][i] for i in ip]
            assert len(bounds) == 2, "contrast for 2 dimensions"
            data = [self.samples_x_argmin[0].take(ip, -1), self.samples_x_argmin[1].take(ip, -1)]
            xlabel = r'$\mathregular{{x_{}}}$'.format(ip[0]+1) 
            ylabel = r'$\mathregular{{x_{}}}$'.format(ip[1]+1)
            title_words = 'x_argmin_{}_{}'.format(*ip)
            contrast_hist_2d(data, label=label, figsize=figsize, bins=bins, bounds=bounds,
            xlabel=xlabel, ylabel=ylabel)
            if set_title:
                plt.gca().set_title(title_words + ' hist')
            plt.tight_layout()
            if self.save:
                plt.savefig(self.path_to + file_name + '-' + title_words + ext)
            plt.clf()
            plt.close()
            if KDE: 
                contrast_KDE_2d(data, label=label, figsize=figsize, bounds=bounds,
                xlabel=xlabel, ylabel=ylabel)
                if set_title:
                    plt.gca().set_title(title_words + ' KDE')
                plt.tight_layout()
                if self.save:
                    plt.savefig(self.path_to + 'KDE-' + file_name + '-' + title_words + ext)
                plt.clf()
                plt.close()

        # fig = plt.figure(figsize=figsize)
        # ax1 = fig.add_subplot(121)
        # hist1 = ax1.hist2d(self.samples_x_argmin[0][:,0], self.samples_x_argmin[0][:,-1], normed=True,\
        # bins=20, range=bounds, norm=LogNorm())
        # fig.colorbar(hist1[-1])
        # ax1.set_title(label[0])
        # ax1.set_xlabel('x1')
        # ax1.set_ylabel('x2')
        
        # ax2 = fig.add_subplot(122)
        # ax2.hist2d(self.samples_x_argmin[1][:,0], self.samples_x_argmin[1][:,-1], normed=True, \
        # bins=20, range=bounds, norm=LogNorm())
        # fig.colorbar(hist1[-1])
        # ax2.set_title(label[1])
        # ax2.set_xlabel('x1')
        # ax2.set_ylabel('x2')

        
