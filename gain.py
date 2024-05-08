import gc
import os
from datetime import datetime

import numpy as np
from scipy.optimize import curve_fit

#import common as cm
#import params as pm

from . import common as cm
from . import params as pm

class Gain(cm.Common):
    def __init__(self, prm_file, filter_dir):
        self.load(prm_file, filter_dir)
        print('Gain object created')

    def load(self, prm_file, filter_dir):
        self.prm = pm.Params(prm_file)
        parameters = self.prm.get_dict()
        #common parameters
        self.results_dir = parameters['common_results_dir']
        self.column_size = parameters['common_column_size']
        self.row_size = parameters['common_row_size']
        self.key_ints = parameters['common_key_ints']
        self.bad_pixels = parameters['common_bad_pixels']

        #gain parameters
        self.nreps = parameters['filter_nreps']
        self.nframes = parameters['filter_nframes']
        self.min_signals = parameters['gain_min_signals']

        print(f'Parameters loaded:')
        self.prm.print_contents()
        
        print('Checking parameters in filter directory')
        #look for a json file in the filter directory
        if (not self.prm.same_common_params(filter_dir)) \
            or (not self.prm.same_offnoi_params(filter_dir) \
            or (not self.prm.same_filter_params(filter_dir))):
            print('Parameters in filter directory do not match')
            return
        try:
            self.event_map = cm.get_array_from_file(
                filter_dir, 'event_map.npy')
            #set the directory where the filter data is stored
            self.filter_dir = filter_dir
            #this is the parent directory. data from this step is stored there
            self.common_dir = os.path.dirname(filter_dir)
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            self.step_dir = os.path.join(
                self.common_dir, timestamp + f'_gain_{self.min_signals}_minsignals'
            )
        except:
            print('Error loading filter data\n')
            return
        print('Filter data loaded\n')

    def calculate(self):
        #create the working directory for the gain step
        self.step_dir = os.path.join(self.common_dir, 
                                     f'gain_{self.min_signals}_min_signals')
        os.makedirs(self.step_dir, exist_ok=True)
        print(f'Created directory for gain step: {self.step_dir}')
        # and save the parameter file there
        self.prm.save(os.path.join(self.step_dir, 'parameters.json'))
        
        fits = self.get_gain_fit(self.event_map)
        np.save(os.path.join(self.step_dir, 'fit_mean.npy'), fits[0])
        np.save(os.path.join(self.step_dir, 'fit_sigma.npy'), fits[1])
        np.save(os.path.join(self.step_dir, 'fit_mean_error.npy'), fits[2])
        np.save(os.path.join(self.step_dir, 'fit_sigma_error.npy'), fits[3])
        
    def get_gain_fit(self, event_map):
        mean = np.full((self.row_size, self.column_size), np.nan)
        sigma = np.full((self.row_size, self.column_size), np.nan)
        mean_error = np.full((self.row_size, self.column_size), np.nan)
        sigma_error = np.full((self.row_size, self.column_size), np.nan)

        def fit_hist(data):

            def gaussian(x, a1, mu1, sigma1):
                return a1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
            try:
                hist, bins = np.histogram(data, bins=10, range=(np.nanmin(data), np.nanmax(data)), density=True)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                params, covar = curve_fit(gaussian, 
                                          bin_centers, 
                                          hist, 
                                          p0=[1,bins[np.argmax(hist)],1])
                return (params[1], abs(params[2]), 
                        np.sqrt(np.diag(covar))[1], 
                        np.sqrt(np.diag(covar))[2])
            except:
                return (np.nan,np.nan,np.nan,np.nan)
        
        count_too_few = 0
        for i in range(event_map.shape[0]):
            for j in range(event_map.shape[1]):
                signals = event_map[i,j]
                if len(signals) >= self.min_signals:
                    params = fit_hist(signals)
                    mean[i,j] = params[1]
                    sigma[i,j] = params[1]
                    mean_error[i,j] = params[2]
                    sigma_error[i,j] = params[3]
                else:
                    count_too_few += 1
        print(f'{count_too_few} pixels have less than {self.min_signals} signals')
        return mean, sigma, mean_error, sigma_error