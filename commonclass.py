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

parameters: dict:
{
    'dark_bin_file': str,
    'signal_bin_file': str,
    # The path to the binary files
    'common_key_ints': int,
    # The number of key integers at the end of a row
    'common_column_size': int,
    # The number of columns of the detector
    'common_row_size': int,
      # The number of rows of the detector
    'common_bad_pixels' : int,
    # Number of bad pixels to be ignored in the offnoi step
    'dark_nreps': int,
    'signal_nreps': int,
    # The number of repetitions
    'dark_nframes': int,
    'signal_nframes': int,
    # The number of frames to be analysed
    'dark_comm_mode' : bool,
    'signal_comm_mode' : bool,
    # Common Mode Correction
    'dark_thres_mips' : int,
    'signal_thres_mips' : int,
    # Threshold for the offnoi step for filtering out the mips
    'signal_thres_event' : int,
    # Threshold for the filter step for filtering out the events
    'signal_fitted_offset' : bool,
}
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

    def read_data(self):
        frames_per_chunk = 20
        raw_row_size = self.column_size + self.key_ints
        raw_frame_size = self.column_size * raw_row_size * self.nreps
        chunk_size =  raw_frame_size* frames_per_chunk
        offset = 8
        count = 0
        self.data = np.zeros((self.nframes, self.column_size, self.nreps, 
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
                break
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
                self.data[frames_here:] = inp_data[:self.nframes-frames_here]
                frames_here += frames_inc
                print(f'\nLoaded {self.nframes} frames')
                print('Run calculate()\n~~~~~')
                break
            #here is the polarity from the c++ code
            self.data[frames_here:frames_here+frames_inc] = \
            -inp_data.reshape(-1, self.column_size, 
                              self.nreps, self.row_size).astype(float)
            frames_here += frames_inc
            gc.collect()

    def _calc_avg_over_frames(self):
        self.avg_over_frames = np.nanmean(self.data, axis = 0)

    def _calc_avg_over_nreps(self):
        self.avg_over_nreps = np.nanmean(self.data, axis = 2)

    def _calc_avg_over_frames_and_nreps(self):
        self.avg_over_frames_and_nreps = np.nanmean(self.data, axis = (0,2))

    def _exclude_mips_frames(self, threshold):
        print(f'Excluding bad frames due to MIPS, threshold: {threshold}')
        median = np.nanmedian(self.data, axis = (1,2,3))[:,np.newaxis,np.newaxis,np.newaxis]
        mask = (self.data > median + threshold) | (self.data < median - threshold)
        del median
        gc.collect()
        mask = np.any(mask, axis = (1,2,3))
        self.data = self.data[~mask]
        print(f'Excluded {np.sum(mask)} frames')
    
    def _set_bad_pixels(self):
        '''
        Sets all ignored Pixels in self.data to NaN.
        Keyword Arguments:
        bad_pixels  --- list of tuples (column,row) of pixels to ignore
        '''
        print('Excluding bad pixels')
        bad_pixel_mask = np.zeros(self.data.shape, dtype=bool)
        for index in self.bad_pixels:
            col = index[0]
            row = index[1]
            bad_pixel_mask[:,row,:,col] = True
            self.data[bad_pixel_mask] = np.nan
        print(f'Excluded {len(self.bad_pixels)} pixels')

    def _correct_common_mode(self):
        '''
        Performs the common mode correction.
        Calculates the median of euch row in data, and substracts it from the row.
        '''
        print('Starting common mode correction')  
        median_common = np.nanmedian(self.data, axis = 3)[:,:,:,np.newaxis]
        self.data = self.data - median_common
        print('Data is corrected for common mode')

    def _get_bin_file_name(self):
        return os.path.basename(self.bin_file)
    
    def _get_fitted_offnoi(self, estimated_mean = 0):
    
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
            
        fitAll = np.apply_along_axis(fitHistOverNreps, axis = 0, arr = self.avg_over_nreps)
        return fitAll[0], fitAll[1], fitAll[2], fitAll[3]

    def fit_gauss_to_hist(data_to_fit):
        return None
    
    def _get_common_dir(self):
        return self.common_dir
    
    def _get_script_dir(self):
        return self.results_dir
    
    def _get_step_dir(self):
        return self.step_dir