import gc
import os
from datetime import datetime

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
from numba import prange
from iminuit import cost, Minuit
from .logger import Logger

'''
This is the base class.
It is a common class that is used by all the other classes in the package.
'''

class Common:
    '''
    there is no constructor here, everything is done in the child classes
    take care with variable names!
    eg: self.nframes must have the same name in every Child class
    '''
    _logger = Logger('nproan-common', 'debug').get_logger()

    def get_data(self):
        '''
        Reads the binary file (in chunks) and returns the data as a numpy array 
        in shape (nframes, column_size, nreps, row_size).

        Args:
            None; uses class variables, that are set through the parameter file
        
        Returns:
            np.array in shape (nframes, column_size, nreps, row_size)
        '''
        polarity = -1
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
                self._logger.warning(f'Loaded {frames_here} of {self.nframes} requested frames, end of file reached.')
                return polarity*output[:frames_here].copy()
            self._logger.info(f'Reading chunk {count+1} from bin file, \
                  {frames_here} frames loaded')
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
                self._logger.error('No valid frames found in chunk, wrong nreps.')
                raise Exception('No valid frames found in chunk, wrong nreps.')
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
                inp_data = inp_data.reshape(-1, self.column_size, 
                                             self.nreps, 
                                             self.row_size).astype(float)
                output[frames_here:] = inp_data[:self.nframes-frames_here]
                frames_here += frames_inc
                self._logger.info(f'\nLoaded {self.nframes} frames')
                return polarity*output.copy()
            output[frames_here:frames_here+frames_inc] = \
            inp_data.reshape(-1, self.column_size, 
                              self.nreps, self.row_size).astype(float)
            frames_here += frames_inc
            gc.collect()
    @staticmethod
    def get_avg_over_frames(data):
        '''
        Calculates the average over the frames in data.

        Args:
            np.array in shape (nframes, column_size, nreps, row_size)
        
        Returns:
            np.array in shape (column_size, nreps, row_size)
        '''
        if np.ndim(data) != 4:
            Common._logger.error('Data has wrong dimensions')
            return None
        return np.nanmean(data, axis = 0)
    @staticmethod
    def get_avg_over_nreps(data):
        '''
        Calculates the average over the nreps in data.

        Args:
            np.array in shape (nframes, column_size, nreps, row_size)

        Returns:
            np.array in shape (nframes, column_size, row_size)
        '''
        if np.ndim(data) != 4:
            Common._logger.error('Data has wrong dimensions')
            return None
        return np.nanmean(data, axis = 2)
    @staticmethod
    def get_avg_over_frames_and_nreps(data):
        '''
        Calculates the average over the frames and nreps in data.

        Args:
            np.array in shape (nframes, column_size, nreps, row_size)

        Returns:
            np.array in shape (column_size, row_size)
        '''
        if np.ndim(data) != 4:
            Common._logger.error('Data has wrong dimensions')
            return None
        return np.nanmean(data, axis = (0,2))

    def exclude_nreps_eval(self, data):
        '''
        Deletes nreps from data that are not in the list nreps_eval.
        nreps_eval is a list of 3 integers: [lower, upper, step]

        Args:
            np.array in shape (nframes, column_size, nreps, row_size)

        Returns:
            np.array in shape (nframes, column_size, nreps-X, row_size)
        '''

        if np.ndim(data) != 4:
            self._logger.error('Data has wrong dimensions')
            return None
        if len(self.nreps_eval) != 3:
            self._logger.error('nreps_eval must be a list of 3 integers')
            raise ValueError('nreps_eval must be a list of 3 integers')
        lower = self.nreps_eval[0]
        upper = self.nreps_eval[1]
        step = self.nreps_eval[2]
        if upper == -1:
            upper = data.shape[2]
        if lower < 0:
            raise ValueError('Lower limit must be greater or equal 0')
        if upper > data.shape[2]:
            raise ValueError('Upper limit is greater than the number of nreps')
        if upper < lower:
            raise ValueError('Upper limit must be greater than lower limit')

        self._logger.info('Excluding nreps')
        mask = np.zeros(data.shape[2])
        mask[lower:upper:step] = 1
        mask = mask.astype(bool)
        self._logger.info(f'Excluded {np.sum(~mask)} nreps')
        return data[:,:,mask,:]

    def exclude_mips_frames(self, data):
        '''
        Calculates the median of each frame and deletes frames that are
        above or below the median by a certain threshold.

        Args:
            np.array in shape (nframes, column_size, nreps, row_size)

        Returns:
            np.array in shape (nframes-X, column_size, nreps, row_size)
        '''
        if np.ndim(data) != 4:
            self._logger.error('Data has wrong dimensions')
            return None
        self._logger.info(f'Excluding bad frames due to MIPS, threshold: {self.thres_mips}')
        median = np.nanmedian(data, 
                              axis = (1,2,3))[:,np.newaxis,np.newaxis,np.newaxis]
        mask = (data > median + self.thres_mips) | (data < median - self.thres_mips)
        del median
        gc.collect()
        mask = np.any(mask, axis = (1,2,3))
        self._logger.info(f'Excluded {np.sum(mask)} frames')
        return data[~mask]

    def exclude_bad_frames(self, data):
        '''
        Calculates the average of each frame and excludes frames that are
        above or below the fitted mean by a certain threshold.
        It saves a .png file in the step directory.

        Args:
            np.array in shape (nframes, column_size, nreps, row_size)

        Returns:
            np.array in shape (nframes-X, column_size, nreps, row_size)
        '''
        if np.ndim(data) != 4:
            self._logger.error('Data has wrong dimensions')
            return None
        self._logger.info('Excluding bad frames')
        avg_per_frame = np.nanmean(data, axis = (1,2,3))
        np.save(os.path.join(self.step_dir, 'avg_per_frame.npy'), avg_per_frame)
        fit = fit_gauss_to_hist(avg_per_frame)
        lower_bound = fit[1] - self.thres_bad_frames*np.abs(fit[2])
        upper_bound = fit[1] + self.thres_bad_frames*np.abs(fit[2])
        bad_pixel_mask = (avg_per_frame < lower_bound) | (avg_per_frame > upper_bound)
        excluded_frames = np.sum(bad_pixel_mask)
        self._logger.info(f'Excluded {excluded_frames} frames')
        title = f'Average signal per frame. Excluded {excluded_frames} frames'
        draw_hist_and_gauss_fit(avg_per_frame, 100, fit[0], fit[1], fit[2],
                                'bad_frames', 
                                save_to = self.step_dir)
        return data[~bad_pixel_mask]
    
    def get_bad_slopes(self, data):
        '''
        Calculates the slope over nreps for every pixel and frame.
        It then fits a gaussian to the histogram of the slopes, and determines
        the bad slopes by a threshold.
        
        Args:
            np.array in shape (nframes, column_size, nreps, row_size)
            
        Returns:
            np.array in shape (n, 3) with the position [frame, row, column]
            np.array in shape (n, nreps) with the data of the bad slopes
        '''
        if np.ndim(data) != 4:
            self._logger.error('Data has wrong dimensions')
            return None
        self._logger.info('Calculating bad slopes')
        slopes = np.apply_along_axis(linear_fit, axis = 2, arr = data)[:, :, 0, :]
        self._logger.debug(f'Shape of slopes: {slopes.shape}')
        fit = fit_gauss_to_hist(slopes.flatten())
        self._logger.debug(f'Fit: {fit}')
        lower_bound = fit[1] - self.thres_bad_slopes*np.abs(fit[2])
        upper_bound = fit[1] + self.thres_bad_slopes*np.abs(fit[2])
        self._logger.debug(f'Lower bound: {lower_bound}, Upper bound: {upper_bound}')
        bad_slopes_mask = (slopes < lower_bound) | (slopes > upper_bound)
        frame, row, column = np.where(bad_slopes_mask)
        bad_slopes_pos = np.array([frame, row, column]).T
        bad_slopes_value = slopes[bad_slopes_mask]
        #get indices of frames with bad slopes
        bad_slopes_data = data[frame.T, row.T, :, column.T]
        self._logger.info(f'Found {len(bad_slopes_pos)} bad Slopes')
        self._logger.debug(f'Shape of bad slopes data: {bad_slopes_data.shape}')
        self._logger.infdebugo(f'Shape of bad slopes pos: {bad_slopes_pos.shape}')
        title = f'Slope Values for each Pixel and Frame. {len(bad_slopes_pos)} bad slopes found.'
        draw_hist_and_gauss_fit(slopes.flatten(), 100, fit[0], fit[1], fit[2],
                                'bad_slopes', 
                                save_to = self.step_dir)
        return bad_slopes_pos, bad_slopes_data, bad_slopes_value

    def set_bad_pixellist_to_nan(self, data):
        '''
        Sets all ignored Pixels in data to NaN. List of pixels is from the
        parameter file. [(row,col), (row,col), ...]

        Args:
            np.array in shape (nframes, column_size, nreps, row_size)
        
        Returns:
            np.array in shape (nframes, column_size, nreps, row_size)
        '''
        if np.ndim(data) != 4:
            self._logger.error('Data has wrong dimensions')
            return None
        self._logger.info('Excluding bad pixels')
        bad_pixel_mask = np.zeros(data.shape, dtype=bool)
        for index in self.bad_pixels:
            col = index[1]
            row = index[0]
            bad_pixel_mask[:,row,:,col] = True
        data[bad_pixel_mask] = np.nan
        self._logger.info(f'Excluded {len(self.bad_pixels)} pixels')
        return data

    def correct_common_mode(self,data):
        '''
        Calculates the median of euch row in data, and substracts it from 
        the row.
        Correction is done inline to save memory.

        Args:
            np.array in shape (nframes, column_size, nreps, row_size)
        '''
        self._logger.info(f'Starting common mode correction at {datetime.now()}')  
        # Iterate over the nframes dimension
        for i in range(data.shape[0]):
            # Calculate the median for one frame
            median_common = np.nanmedian(data[i], axis=2, keepdims=True)
            # Subtract the median from the frame in-place
            data[i] -= median_common
        self._logger.info(f'Data is corrected for common mode at {datetime.now()}')
    
    def get_bin_file_name(self):
        return os.path.basename(self.bin_file)

    def get_common_dir(self):
        return self.common_dir
    
    def get_script_dir(self):
        return self.results_dir
    
    def get_step_dir(self):
        return self.step_dir
    
def get_unbinned_fit_gauss(data):
    '''
    fits a gaussian to a histogram of data
    using the unbinned method in minuit
    returns a np.array in shape (6, rows, columns)

    Args:
        np.array in shape (nframes, column_size, row_size)

    Returns:
        np.array in shape (6, rows, columns)
        index 0: amplitude
        index 1: mean
        index 2: sigma
        index 3: error_amplitude
        index 4: error_mean
        index 5: error_sigma
    '''
    if data.ndim != 3:
        Common._logger.error('Data has wrong dimensions')
        return
    #apply the function to every frame
    output = np.apply_along_axis(unbinned_fit_gauss_to_hist, axis = 0, arr = data)
    return output

def get_fit_gauss(data):
    ''' 
    fits a gaussian to a histogram of data_to_fit
    using the scipy curve_fit method

    Args:
        np.array in shape (nframes, column_size, row_size)

    Returns:
        np.array in shape (6, rows, columns)
        index 0: amplitude
        index 1: mean
        index 2: sigma
        index 3: error_amplitude
        index 4: error_mean
        index 5: error_sigma
    '''
    if data.ndim != 3:
        Common._logger.error('Data has wrong dimensions')
        return
    #apply the function to every frame
    output = np.apply_along_axis(fit_gauss_to_hist, axis = 0, arr = data)
    return output

def fit_gauss_to_hist(data_to_fit):
    ''' 
    fits a gaussian to a histogram of data_to_fit

    Args:
        1D np.array
    
    Returns:
        np.array[amplitude, mean, sigma, error_amplitude, error_mean, error_sigma]
    '''
    guess = [1, np.nanmedian(data_to_fit), np.nanstd(data_to_fit)]
    try:
        hist, bins = np.histogram(data_to_fit, bins=100, range=(np.nanmin(data_to_fit), np.nanmax(data_to_fit)), density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        params, covar = curve_fit(gaussian, bin_centers, hist, p0=guess)
        return np.array([params[0],
                         params[1], 
                         np.abs(params[2]),
                         np.sqrt(np.diag(covar))[0],
                         np.sqrt(np.diag(covar))[1], 
                         np.sqrt(np.diag(covar))[2]])
    except:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

def unbinned_fit_gauss_to_hist(data_to_fit):
        '''
        fits a gaussian to a histogram of data_to_fit

        Args:
            1D np.array

        Returns:
            np.array[amplitude, mean, sigma, error_mean, error_sigma]
        '''
        #TODO: this doesnt seem to work: test this!
        c = cost.UnbinnedNLL(data_to_fit, gaussian)
        m = Minuit(c, 
                   a1=1, 
                   mu1=np.nanmedian(data_to_fit), 
                   sigma1=np.nanstd(data_to_fit))
        m.limits['a1'] = (0, 100)
        m.limits['mu1'] = (np.nanmin(data_to_fit), np.nanmax(data_to_fit))
        m.limits['sigma1'] = (0, 100)
        m.migrad()
        if not m.valid:
            return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        return np.array([m.values['a1'],
                         m.values['mu1'],
                         np.abs(m.values['sigma1']),
                         m.errors['a1'],
                         m.errors['mu1'],
                         m.errors['sigma1']])

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
        Common._logger.error('Data is not 2D')
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
    plt.plot(bin_centers, gaussian(bin_centers, amplitude, mean, sigma), 'r-')
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

def get_array_from_file(folder, filename):
    return np.load(os.path.join(folder, filename), allow_pickle=True)

def gaussian(x, a1, mu1, sigma1):
    return (a1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2)))

def two_gaussians(x, a1, mu1, sigma1, a2, mu2, sigma2):
    return (a1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2) +
            a2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))))

def linear_fit(data):
    x = np.arange(data.size)
    k, d = np.polyfit(x, data, 1)
    return np.array([k, d])

def get_pixels_to_nan(data, indices):
    '''
    copies the data array, sets all pixels to nan
    
    Args:
        np.array in shape (nframes, column_size, row_size)
        list of tuples [(frame, row, column), (frame, row, column), ...]
    '''
    if np.ndim(data) != 3:
        Common._logger.error('Data has wrong dimensions')
        return None
    if np.ndim(indices) != 2:
        Common._logger.error('Pixel positions have wrong dimensions')
        return None
    data_copy = data.copy()
    #TODO: Vectorize this
    for entry in indices:
        data_copy[entry[0], entry[1], entry[2]] = np.nan
    return data_copy

def get_rolling_average(data, window_size):
    '''
    Calculates a rolling average over window_size
    
    Args:
        1D np.array
        window_size: int
    
    Returns:
        1D np.array
    '''
    weights = np.repeat(1.0, window_size) / window_size
    # Use 'valid' mode to ensure that output has the same length as input
    return np.convolve(data, weights, mode='valid')

def load_npy_files(folder):
    # Get a list of all .npy files in the folder
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]

    # Load each file into a numpy array and store it in a dictionary
    arrays = {}
    for file in files:
        # Remove the .npy extension from the filename
        name = os.path.splitext(file)[0]
        # Load the file and store it in the dictionary
        arrays[name] = np.load(os.path.join(folder, file), allow_pickle=True)

    return arrays

def sort_with_indices(arr):
    '''
    Sorts array in descending order and returns the indices.

    Args:
        1D np.array

    Returns:
        1D np.array
    '''
    indexed_arr = np.column_stack((np.arange(len(arr)), arr))
    sorted_indices = indexed_arr[np.argsort(indexed_arr[:, 1])[::-1]][:, 0]
    return sorted_indices.astype(int)