import gc
import os
from datetime import datetime

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
from iminuit import cost, Minuit

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
    def get_data(self):
        '''reads the binary file and returns the data as a numpy array.
        It is read one chunk at a time, to save memory.
        The data is reshaped into the dimensions 
        (nframes, column_size, nreps, row_size)'''
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
                print(f'\nLoaded {frames_here} of {self.nframes} requested frames, end of file reached.')
                print('Run calc()\n~~~~~')
                return output[:frames_here].copy()
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
                inp_data = inp_data.reshape(-1, self.column_size, 
                                             self.nreps, 
                                             self.row_size).astype(float)
                output[frames_here:] = inp_data[:self.nframes-frames_here]
                frames_here += frames_inc
                print(f'\nLoaded {self.nframes} frames')
                print('Run calculate()\n~~~~~')
                return output
            #here is the polarity from the c++ code
            output[frames_here:frames_here+frames_inc] = \
            inp_data.reshape(-1, self.column_size, 
                              self.nreps, self.row_size).astype(float)
            frames_here += frames_inc
            gc.collect()

    @staticmethod
    def get_avg_over_frames(data):
    #TODO: move this to outside of class
        if np.ndim(data) != 4:
            print('Data has wrong dimensions')
            return None
        return np.nanmean(data, axis = 0)

    @staticmethod
    def get_avg_over_nreps(data):
    #TODO: move this to outside of class
        if np.ndim(data) != 4:
            print('Data has wrong dimensions')
            return None
        return np.nanmean(data, axis = 2)

    @staticmethod
    def get_avg_over_frames_and_nreps(data):
    #TODO: move this to outside of class
        if np.ndim(data) != 4:
            print('Data has wrong dimensions')
            return None
        return np.nanmean(data, axis = (0,2))

    def exclude_nreps_eval(self, data):
        '''Excludes nreps from data that are not in the list nreps_eval.
        Returns the data without the excluded nreps'''

        if np.ndim(data) != 4:
            print('Data has wrong dimensions')
            return None
        if len(self.nreps_eval) != 3:
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

        print('Excluding nreps')
        mask = np.zeros(data.shape[2])
        mask[lower:upper:step] = 1
        mask = mask.astype(bool)
        print(f'Excluded {np.sum(~mask)} nreps')
        return data[:,:,mask,:]

    def exclude_mips_frames(self, data):
        '''Calculates the median of each frame and excludes frames that are
        above or below the median by a certain threshold.
        returns the data without the bad frames'''
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
        return data[~mask]

    def exclude_bad_frames(self, data):
        '''Calculates the average of each frame and excludes frames that are
        above or below the fitted mean by a certain threshold.
        It saves a .png file in the step directory.
        returns the data without the bad frames'''
        if np.ndim(data) != 4:
            print('Data has wrong dimensions')
            return None
        print('Excluding bad frames')
        avg_per_frame = np.nanmean(data, axis = (1,2,3))
        np.save(os.path.join(self.step_dir, 'avg_per_frame.npy'), avg_per_frame)
        fit = fit_gauss_to_hist(avg_per_frame)
        lower_bound = fit[1] - self.thres_bad_frames*np.abs(fit[2])
        upper_bound = fit[1] + self.thres_bad_frames*np.abs(fit[2])
        bad_pixel_mask = (avg_per_frame < lower_bound) | (avg_per_frame > upper_bound)
        excluded_frames = np.sum(bad_pixel_mask)
        print(f'Excluded {excluded_frames} frames')
        title = f'Average signal per frame. Excluded {excluded_frames} frames'
        draw_hist_and_gauss_fit(avg_per_frame, 100, fit[0], fit[1], fit[2],
                                title, 'bad_frames', 
                                save_to = self.step_dir)
        return data[~bad_pixel_mask]
    
    def get_bad_slopes(self, data):
        if np.ndim(data) != 4:
            print('Data has wrong dimensions')
            return None
        print('Calculating bad slopes')
        slopes = np.apply_along_axis(linear_fit, axis = 2, arr = data)[:, :, 0, :]
        fit = fit_gauss_to_hist(slopes.flatten())
        lower_bound = fit[1] - self.thres_bad_frames*np.abs(fit[2])
        upper_bound = fit[1] + self.thres_bad_frames*np.abs(fit[2])
        bad_slopes_mask = (slopes < lower_bound) | (slopes > upper_bound)
        frame, row, column = np.where(bad_slopes_mask)
        bad_slopes_pos = np.array([frame, column, row]).T
        #get indices of frames with bad slopes
        bad_slopes_data = data[frame, row, :, column]
        print(f'Found {len(bad_slopes_pos)} bad Slopes')
        return bad_slopes_pos, bad_slopes_data

    def set_bad_pixels_to_nan(self, data):
        '''Sets all ignored Pixels in data to NaN.
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
        return data

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

    def get_common_dir(self):
        return self.common_dir
    
    def get_script_dir(self):
        return self.results_dir
    
    def get_step_dir(self):
        return self.step_dir
    
def get_unbinned_fit_gauss(data):
    ''' fits a gaussian to a histogram of data_to_fit
    using the unbinned method in minuit
    returns a np.array in shape (6, rows, columns)
    index 0: amplitude
    index 1: mean
    index 2: sigma
    index 3: error_amplitude
    index 4: error_mean
    index 5: error_sigma
    '''
    if data.ndim != 3:
        print('Data has wrong dimensions')
        return
    #apply the function to every frame
    output = np.apply_along_axis(unbinned_fit_gauss_to_hist, axis = 0, arr = data)
    return output

def get_fit_gauss(data):
    ''' fits a gaussian to a histogram of data_to_fit
    using the scipy curve_fit method
    returns a np.array in shape (6, rows, columns)
    index 0: amplitude
    index 1: mean
    index 2: sigma
    index 3: error_amplitude
    index 4: error_mean
    index 5: error_sigma
    '''
    if data.ndim != 3:
        print('Data has wrong dimensions')
        return
    #apply the function to every frame
    output = np.apply_along_axis(fit_gauss_to_hist, axis = 0, arr = data)
    return output

def fit_gauss_to_hist(data_to_fit):
    ''' fits a gaussian to a histogram of data_to_fit
    return np.array[amplitude, mean, sigma, error_mean, error_sigma]
    '''
    guess = [1, np.median(data_to_fit), np.std(data_to_fit)]
    try:
        hist, bins = np.histogram(data_to_fit, bins=100, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        params, covar = curve_fit(gaussian, bin_centers, hist, p0=guess)
        return np.array([params[0],
                         params[1], 
                         params[2],
                         np.sqrt(np.diag(covar))[0],
                         np.sqrt(np.diag(covar))[1], 
                         np.sqrt(np.diag(covar))[2]])
    except:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])

def unbinned_fit_gauss_to_hist(data_to_fit):
        ''' fits a gaussian to a histogram of data_to_fit
        return np.array[amplitude, mean, sigma, error_mean, error_sigma]
        '''
        #TODO: this doesnt seem to work: test this!
        c = cost.UnbinnedNLL(data_to_fit, gaussian)
        m = Minuit(c, 
                   a1=1, 
                   mu1=np.median(data_to_fit), 
                   sigma1=np.std(data_to_fit))
        m.limits['a1'] = (0, 100)
        m.limits['mu1'] = (np.min(data_to_fit), np.max(data_to_fit))
        m.limits['sigma1'] = (0, 100)
        m.migrad()
        if not m.valid:
            return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        return np.array([m.values['a1'],
                         m.values['mu1'],
                         m.values['sigma1'],
                         m.errors['a1'],
                         m.errors['mu1'],
                         m.errors['sigma1']])

def draw_hist(data,file_name="histogram", save_to=None, **kwargs):
    '''
    Draw a histogram of the data
    if save is True, the histogram is saved as a .png file
    kwargs are passed to plt.hist
    '''
    plt.hist(data.ravel(), **kwargs)
    if save_to is not None:
        plt.savefig(save_to + file_name + '.png')
    else:
        plt.show()

def draw_heatmap(data,file_name="heatmap", save_to=None, **kwargs):
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
    if save_to is not None:
        plt.savefig(save_to + file_name + '.png')
    else:
        plt.show()

def draw_graph(data,file_name="graph", save_to=None, **kwargs):
    '''
    Draw a graph of the data
    if save is True, the graph is saved as a .png file
    kwargs are passed to plt.plot
    '''
    plt.plot(data.ravel(), **kwargs)
    if save_to is not None:
        plt.savefig(save_to + file_name + '.png')
    else:
        plt.show()

def draw_hist_and_gauss_fit(data, bins, amplitude, mean, sigma, title, file_name, save_to=None):
    
    hist, hist_bins = np.histogram(data, bins, density=True)
    bin_centers = (hist_bins[:-1] + hist_bins[1:]) / 2
    plt.hist(data, bins=hist_bins, density=True, alpha=0.5, label='Histogram')
    plt.plot(bin_centers, gaussian(bin_centers, amplitude, mean, sigma), 'r-', label='Fitted Curve')
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    if save_to is not None:
        plt.savefig(save_to + file_name + '.png')
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
    m, b = np.polyfit(x, data, 1)
    return np.array([m, b])