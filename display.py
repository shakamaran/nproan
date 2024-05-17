import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import logger
import fitting

_logger = logger.get_logger(__name__, 'info')

def draw_hist(data, file_name="histogram", save_to=None, **kwargs):
    '''
    Draw a histogram of the data
    
    Args:
        np.array
        file_name: str
        save_to: str
        **kwargs: passed to plt.hist
    '''
    plt.clf()
    plt.hist(data.ravel(), **kwargs)
    if save_to is not None:
        plt.savefig(save_to + file_name + '.png')
        plt.close()
    else:
        plt.show()

def draw_heatmap(data, file_name="heatmap", save_to=None, **kwargs):
    '''
    Draw a heatmap of the data
    
    Args:
        np.array
        file_name: str
        save_to: str (optional)
        **kwargs: passed to sns.heatmap
    '''
    if data.ndim !=2:
        _logger.error('Data is not 2D')
        return
    plt.clf()
    # Define default values
    cmap = kwargs.get('cmap', 'coolwarm')
    sns.heatmap(data, cmap=cmap, **kwargs)
    if save_to is not None:
        plt.savefig(save_to + file_name + '.png')
        plt.close()
    else:
        plt.show()

def draw_graph(data, file_name="graph", save_to=None, **kwargs):
    '''
    Draw a graph of the data
    
    Args:
        np.array
        file_name: str
        save_to: str (optional)
        **kwargs: passed to plt.plot
    '''
    plt.clf()
    plt.plot(data.ravel(), **kwargs)
    if save_to is not None:
        plt.savefig(save_to + file_name + '.png')
        plt.close()
    else:
        plt.show()

def draw_hist_and_gauss_fit(data, bins, amplitude, mean, sigma, file_name=None, save_to=None):
    '''
    Draw a histogram of the data and a gaussian fit

    Args:
        np.array
        bins: int
        amplitude: float
        mean: float
        sigma: float
        title: str
        file_name: str
        save_to: str (optional)
    '''
    plt.clf()
    hist, hist_bins = np.histogram(data, bins, range=(np.nanmin(data), np.nanmax(data)), density=True)
    bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
    plt.hist(data, bins=hist_bins, density=True, alpha=0.5)
    plt.plot(bin_centers, fitting.gaussian(bin_centers, amplitude, mean, sigma), 'r-')
    plt.title(f'Fitted parameters:\nMean: {mean:.2f}\nSigma: {sigma:.2f}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    if save_to is not None:
        if file_name is None:
            file_name = 'hist_and_fit'
        plt.savefig(save_to + '/' + file_name + '.png')
        plt.close()
    else:
        plt.show()