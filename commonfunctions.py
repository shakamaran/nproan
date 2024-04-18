import gc
import os
from datetime import datetime

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

def draw_hist(data, save=False, **kwargs):
    '''
    Draw a histogram of the data
    if save is True, the histogram is saved as a .png file
    kwargs are passed to plt.hist
    '''
    plt.hist(data.ravel(), **kwargs)
    if save:
        plt.savefig('histogram.png')
    plt.show()

def draw_heatmap(data, save=False, **kwargs):
    '''
    Draw a heatmap of the data
    if save is True, the heatmap is saved as a .png file
    Returns nothing if data is not 2D
    kwargs are passed to sns.heatmap
    '''
    if data.ndim !=2:
        print('Data is not 2D')
        return
    # Define default values
    cmap = kwargs.get('cmap', 'coolwarm')
    sns.heatmap(data, cmap=cmap, **kwargs)
    if save:
        plt.savefig('heatmap.png')
    plt.show()

def draw_graph(data, save=False, **kwargs):
    '''
    Draw a graph of the data
    if save is True, the graph is saved as a .png file
    kwargs are passed to plt.plot
    '''
    plt.plot(data.ravel(), **kwargs)
    if save:
        plt.savefig('graph.png')
    plt.show()