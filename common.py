import gc
import os
from datetime import datetime

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

'''
This is the base class.
It is a common class that is used by all the other classes in the package.

TODO:
- add more documentation
- write wiki
- write functions for fitting histograms using minuit

My current philosophy for doing stuff:
- Parameters from the parameter file are stored in the class variables
- parameter file is in JSON format
- for every step class, the whole parameter file must be loaded
- the filter class loads the offnoi data, and checks consistency with the
    parameter file
- the gain class loads the filter data, and checks consistency with the
    parameter file
- one directory (the "common_dir") should be created in a "results" directory 
    on \scratch this should only be done once in the offnoi step
    in there, a directory "offnoi" should be created
    the parameter file must be placed there after calculation
- in the filter step a folder path to the "offnoi" directory should be
    provided along with the parameters
    consistency of the parameters (from last steps) will be checked
- in the gain step a folder path to the "filter" directory should be
    provided along with the parameters
    consistency of the parameters (from last steps) will be checked

- functions that return values should be named get_something(), this is 
    preferred
- data should not be stored in the class
- data that is loaded in the filter or gain step is stored in a class variable
    and deleted after use
'''

class Common:
    def __init__(self):
        # these variables are set in the parameters dictionary
        self.bin_file = None
        self.column_size = None
        self.row_size  = None
        self.nreps  = None
        self.key_ints  = None
        self.nframes  = None
        self.bad_pixels = None
        self.comm_mode = False
        self.thres_mips = None

        #the directory where the notebook/script is run
        self.results_dir = os.getcwd()
        #directory with subfolders for each step
        self.common_dir = None
        #directory for the current step
        self.step_dir = None

    def get_data(self):
        frames_per_chunk = 20
        raw_row_size = self.column_size + self.key_ints
        raw_frame_size = self.column_size * raw_row_size * self.nreps
        chunk_size =  raw_frame_size* frames_per_chunk
        offset = 8
        count = 0
        output = np.zeros((self.nframes, self.column_size, self.nreps, 
                              self.row_size), dtype=np.float64)
        frames_here = 0

        while True:
            inp_data = np.fromfile(self.bin_file, dtype='uint16', 
                                   count = chunk_size, offset = offset)
            #check if file is at its end
            if inp_data.size == 0:
                print(f'Loaded {frames_here} of {self.nframes} \
                      requested frames, end of file reached.')
                print('Run calc()\n~~~~~')
                return output
            print(f'\rReading chunk {count+1} from bin file, \
                  {frames_here} frames loaded', end='')
            #reshape the array into rows -> (#ofRows,67)
            inp_data = inp_data.reshape(-1, raw_row_size)
            #find all the framekeys
            frame_keys = np.where(inp_data[:, self.column_size] == 65535)
            #stack them and calculate difference to find incomplete frames
            frames = np.stack((frame_keys[0][:-1], frame_keys[0][1:]))
            del frame_keys
            gc.collect()
            diff = np.diff(frames, axis = 0)
            rows_per_frame = self.row_size*self.nreps
            valid_frames_position = np.nonzero(diff == rows_per_frame)[1]
            if len(valid_frames_position) == 0:
                print('No valid frames found in chunk, maybe nreps wrong?')
                print(f'Here is a hint: {diff}')
                break
            del diff
            gc.collect()
            valid_frames = frames.T[valid_frames_position]
            frame_start_indices = valid_frames[:,0]
            frame_end_indices = valid_frames[:,1]
            del valid_frames
            gc.collect()
            offset += chunk_size * 2
            count += 1
            inp_data = np.array([inp_data[start+1:end+1, :64] 
                                 for start, end 
                                 in zip(frame_start_indices, 
                                        frame_end_indices)])
            frames_inc = inp_data.shape[0]
            #check if inp_data would make self.data full
            if frames_here + frames_inc > self.nframes:
                inp_data = -inp_data.reshape(-1, self.column_size, 
                                             self.nreps, 
                                             self.row_size).astype(float)
                output[frames_here:] = inp_data[:self.nframes-frames_here]
                frames_here += frames_inc
                print(f'\nLoaded {self.nframes} frames')
                print('Run calculate()\n~~~~~')
                return output
            #here is the polarity from the c++ code
            output[frames_here:frames_here+frames_inc] = \
            -inp_data.reshape(-1, self.column_size, 
                              self.nreps, self.row_size).astype(float)
            frames_here += frames_inc
            gc.collect()

    @staticmethod
    def get_avg_over_frames(data):
        if np.ndim(data) != 4:
            print('Data has wrong dimensions')
            return None
        return np.nanmean(data, axis = 0)

    @staticmethod
    def get_avg_over_nreps(data):
        if np.ndim(data) != 4:
            print('Data has wrong dimensions')
            return None
        return np.nanmean(data, axis = 2)

    @staticmethod
    def get_avg_over_frames_and_nreps(data):
        if np.ndim(data) != 4:
            print('Data has wrong dimensions')
            return None
        return np.nanmean(data, axis = (0,2))

    def exclude_mips_frames(self, data):
        if np.ndim(data) != 4:
            print('Data has wrong dimensions')
            return None
        print(f'Excluding bad frames due to MIPS, threshold: {self.thres_mips}')
        median = np.nanmedian(data, 
                              axis = (1,2,3))[:,np.newaxis,np.newaxis,np.newaxis]
        mask = (data > median + self.thres_mips) | (data < median - self.thres_mips)
        del median
        gc.collect()
        mask = np.any(mask, axis = (1,2,3))
        print(f'Excluded {np.sum(mask)} frames')
        data = data[~mask]
    
    def set_bad_pixels_to_nan(self, data):
        '''
        Sets all ignored Pixels in self.data to NaN (inplace).
        Keyword Arguments:
        bad_pixels  --- list of tuples (column,row) of pixels to ignore
        '''
        if np.ndim(data) != 4:
            print('Data has wrong dimensions')
            return None
        print('Excluding bad pixels')
        bad_pixel_mask = np.zeros(data.shape, dtype=bool)
        for index in self.bad_pixels:
            col = index[0]
            row = index[1]
            bad_pixel_mask[:,row,:,col] = True
        data[bad_pixel_mask] = np.nan
        print(f'Excluded {len(self.bad_pixels)} pixels')

    def get_common_corrected_data(self, data):
        '''
        Performs the common mode correction.
        Calculates the median of euch row in data, and substracts it from the row.
        '''
        if np.ndim(data) != 4:
            print('Data has wrong dimensions')
            return None
        print('Starting common mode correction')  
        median_common = np.nanmedian(data, axis = 3)[:,:,:,np.newaxis]
        print('Data is corrected for common mode')
        return data - median_common

    def get_bin_file_name(self):
        return os.path.basename(self.bin_file)
    
    def get_fitted_offnoi(self, data, estimated_mean = 0):
        if np.ndim(data) != 3:
            print('Data has wrong dimensions')
            return None
        
        def fitHistOverNreps(data_to_fit):

            def gaussian(x, a1, mu1, sigma1):
                return a1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
    
            # Initial guesses for parameters
            initial_guesses = [1, estimated_mean, 1]
            try:
                hist, bins = np.histogram(data_to_fit, bins=100, density=True)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                params, covar = curve_fit(
                    gaussian, bin_centers, hist, p0=initial_guesses
                )
                #(mean, std_dev, error_mean, error_std_dev)
                return (params[1],
                        abs(params[2]), 
                        np.sqrt(np.diag(covar))[1], 
                        np.sqrt(np.diag(covar))[2])
            except:
                return (np.nan,np.nan,np.nan,np.nan)
            
        fitAll = np.apply_along_axis(fitHistOverNreps, axis = 0, arr = data)
        return fitAll[0], fitAll[1], fitAll[2], fitAll[3]

    def fit_gauss_to_hist(data_to_fit):
        return None
    
    def get_common_dir(self):
        return self.common_dir
    
    def get_script_dir(self):
        return self.results_dir
    
    def get_step_dir(self):
        return self.step_dir
    

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

def draw_hist_and_fit(data, bins, mean, sigma, pixel_row, pixel_column):
    
    def gaussian(x, a1, mu1, sigma1):
        return (a1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)))
    
    hist, hist_bins = np.histogram(data, bins, density=True)
    bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
    plt.hist(data, bins=hist_bins, density=True, alpha=0.5, label='Histogram')
    plt.plot(bin_centers, gaussian(bin_centers, 1, mean, sigma), 'r-', label='Fitted Curve')
    plt.title(f'Fitted histogram for pixel ({pixel_row},{pixel_column})')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def get_array_from_file(folder, filename):
    return np.load(os.path.join(folder, filename), allow_pickle=True)

def gaussian(x, a1, mu1, sigma1):
    return (a1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)))

def two_gaussians(x, a1, mu1, sigma1, a2, mu2, sigma2):
    return (a1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2) +
            a2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))))