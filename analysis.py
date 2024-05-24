import gc

import numpy as np
from scipy.optimize import curve_fit

from . import logger
from . import fitting
from . import display

_logger = logger.Logger(__name__, 'info').get_logger()

def get_data(bin_file, column_size, row_size, key_ints, nreps, nframes):
        '''
        Reads the binary file (in chunks) and returns the data as a numpy array 
        in shape (nframes, column_size, nreps, row_size).

        Args:
            bin_file: path to the binary file
            column_size: number of columns in the data
            row_size: number of rows in the data
            key_ints: number of key ints in the data
            nreps: number of repetitions in the data
            nframes: number of frames to read from the data
        
        Returns:
            np.array in shape (nframes, column_size, nreps, row_size)
        '''
        polarity = -1
        frames_per_chunk = 20
        raw_row_size = column_size + key_ints
        raw_frame_size = column_size * raw_row_size * nreps
        chunk_size =  raw_frame_size* frames_per_chunk
        offset = 8
        count = 0
        output = np.zeros((nframes, column_size, nreps, 
                              row_size), dtype=np.float64)
        frames_here = 0

        while True:
            inp_data = np.fromfile(bin_file, dtype='uint16', 
                                   count = chunk_size, offset = offset)
            #check if file is at its end
            if inp_data.size == 0:
                _logger.warning(f'Loaded {frames_here} of {nframes} requested frames, end of file reached.')
                return polarity*output[:frames_here].copy()
            _logger.info(f'Reading chunk {count+1} from bin file, \
                  {frames_here} frames loaded')
            #reshape the array into rows -> (#ofRows,67)
            inp_data = inp_data.reshape(-1, raw_row_size)
            #find all the framekeys
            frame_keys = np.where(inp_data[:, column_size] == 65535)
            #stack them and calculate difference to find incomplete frames
            frames = np.stack((frame_keys[0][:-1], frame_keys[0][1:]))
            del frame_keys
            gc.collect()
            diff = np.diff(frames, axis = 0)
            rows_per_frame = row_size*nreps
            valid_frames_position = np.nonzero(diff == rows_per_frame)[1]
            if len(valid_frames_position) == 0:
                _logger.error('No valid frames found in chunk, wrong nreps.')
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
            #check if inp_data would make data full
            if frames_here + frames_inc > nframes:
                inp_data = inp_data.reshape(-1, column_size, 
                                             nreps, 
                                             row_size).astype(float)
                output[frames_here:] = inp_data[:nframes-frames_here]
                frames_here += frames_inc
                _logger.info(f'\nLoaded {nframes} frames')
                return polarity*output.copy()
            output[frames_here:frames_here+frames_inc] = \
            inp_data.reshape(-1, column_size, 
                              nreps, row_size).astype(float)
            frames_here += frames_inc
            gc.collect()

def get_dummy_data(column_size, row_size, nreps, nframes):
    '''
    Returns random values for tests.
    Args:
        column_size: number of columns in the data
        row_size: number of rows in the data
        key_ints: number of key ints in the data
        nreps: number of repetitions in the data
        nframes: number of frames to read from the data
        
    Returns:
        np.array in shape (nframes, column_size, nreps, row_size)
    '''
    return np.random.rand(nframes, row_size, nreps, column_size)

def exclude_nreps_eval(data, nreps_eval):
    '''
    Deletes nreps from data that are not in the list nreps_eval.
    nreps_eval is a list of 3 integers: [lower, upper, step]
    Args:
        data: np.array in shape (nframes, column_size, nreps, row_size)
        nreps_eval: list of 3 ints
    Returns:
        np.array in shape (nframes, column_size, nreps-X, row_size)
    '''
    #TODO: optimize this
    if np.ndim(data) != 4:
        _logger.error('Data has wrong dimensions')
        return None
    if len(nreps_eval) != 3:
        _logger.error('nreps_eval must be a list of 3 integers')
        raise ValueError('nreps_eval must be a list of 3 integers')
    lower = nreps_eval[0]
    upper = nreps_eval[1]
    step = nreps_eval[2]
    if upper == -1:
        upper = data.shape[2]
    if lower < 0:
        raise ValueError('Lower limit must be greater or equal 0')
    if upper > data.shape[2]:
        raise ValueError('Upper limit is greater than the number of nreps')
    if upper < lower:
        raise ValueError('Upper limit must be greater than lower limit')
    _logger.info('Excluding nreps')
    mask = np.zeros(data.shape[2])
    mask[lower:upper:step] = 1
    mask = mask.astype(bool)
    _logger.info(f'Excluded {np.sum(~mask)} nreps')
    return data[:,:,mask,:]

def exclude_mips_frames(data, thres_mips):
    '''
    Calculates the median of each frame and deletes frames that are
    above or below the median by a certain threshold.
    Args:
        data: np.array in shape (nframes, column_size, nreps, row_size)
        thres_mips: absolute threshold in adu
    Returns:
        np.array in shape (nframes-X, column_size, nreps, row_size)
    '''
    #TODO: optimize this
    if np.ndim(data) != 4:
        _logger.error('Data has wrong dimensions')
        return None
    _logger.info(f'Excluding bad frames due to MIPS, threshold: {thres_mips}')
    median = np.nanmedian(data, 
                          axis = (1,2,3))[:,np.newaxis,np.newaxis,np.newaxis]
    mask = (data > median + thres_mips) | (data < median - thres_mips)
    del median
    gc.collect()
    mask = np.any(mask, axis = (1,2,3))
    _logger.info(f'Excluded {np.sum(mask)} frames')
    return data[~mask]

def exclude_bad_frames(data, thres_bad_frames, step_dir=None):
    '''
    Calculates the average of each frame and excludes frames that are
    above or below the fitted mean by a certain threshold.
    It saves a .png file in the step directory.

    Args:
        data: np.array in shape (nframes, column_size, nreps, row_size)
        thres_bad_frames: used with the fitted sigma do exclude frames
        step_dir (optional): directory in which plot is saved

    Returns:
        np.array in shape (nframes-X, column_size, nreps, row_size)
    '''
    #TODO: optimize this
    if np.ndim(data) != 4:
        _logger.error('Data has wrong dimensions')
        return None
    _logger.info('Excluding bad frames')
    avg_per_frame = np.nanmean(data, axis = (1,2,3))
    fit = fitting.fit_gauss_to_hist(avg_per_frame)
    lower_bound = fit[1] - thres_bad_frames*np.abs(fit[2])
    upper_bound = fit[1] + thres_bad_frames*np.abs(fit[2])
    bad_pixel_mask = (avg_per_frame < lower_bound) | (avg_per_frame > upper_bound)
    excluded_frames = np.sum(bad_pixel_mask)
    _logger.info(f'Excluded {excluded_frames} frames')
    title = f'Average signal per frame. Excluded {excluded_frames} frames'
    display.draw_hist_and_gauss_fit(avg_per_frame, 100, fit[0], fit[1], fit[2],
                            'bad_frames', 
                            save_to = step_dir)
    return data[~bad_pixel_mask]

def get_bad_slopes(data, thres_bad_slopes, step_dir=None):
    '''
    Calculates the slope over nreps for every pixel and frame.
    It then fits a gaussian to the histogram of the slopes, and determines
    the bad slopes by a threshold.
    
    Args:
        data: np.array in shape (nframes, column_size, nreps, row_size)
        thres_bad_slopes: used with the fitted sigma to determine bad slopes
        step_dir (optional): directory in which plot is saved
        
    Returns:
        np.array in shape (n, 3) with the position [frame, row, column]
        np.array in shape (n, nreps) with the data of the bad slopes
    '''
    #TODO: optimize this
    if np.ndim(data) != 4:
        _logger.error('Data has wrong dimensions')
        return None
    _logger.info('Calculating bad slopes')
    slopes = np.apply_along_axis(fitting.linear_fit, axis = 2, arr = data)[:, :, 0, :]
    _logger.debug(f'Shape of slopes: {slopes.shape}')
    fit = fitting.fit_gauss_to_hist(slopes.flatten())
    _logger.debug(f'Fit: {fit}')
    lower_bound = fit[1] - thres_bad_slopes*np.abs(fit[2])
    upper_bound = fit[1] + thres_bad_slopes*np.abs(fit[2])
    _logger.debug(f'Lower bound: {lower_bound}, Upper bound: {upper_bound}')
    bad_slopes_mask = (slopes < lower_bound) | (slopes > upper_bound)
    frame, row, column = np.where(bad_slopes_mask)
    bad_slopes_pos = np.array([frame, row, column]).T
    bad_slopes_value = slopes[bad_slopes_mask]
    #get indices of frames with bad slopes
    bad_slopes_data = data[frame.T, row.T, :, column.T]
    _logger.info(f'Found {len(bad_slopes_pos)} bad Slopes')
    _logger.debug(f'Shape of bad slopes data: {bad_slopes_data.shape}')
    _logger.debug(f'Shape of bad slopes pos: {bad_slopes_pos.shape}')
    title = f'Slope Values for each Pixel and Frame. {len(bad_slopes_pos)} bad slopes found.'
    display.draw_hist_and_gauss_fit(slopes.flatten(), 100, fit[0], fit[1], fit[2],
                            'bad_slopes', 
                            save_to = step_dir)
    return bad_slopes_pos, bad_slopes_data, bad_slopes_value

def set_bad_pixellist_to_nan(data, bad_pixels):
    '''
    Sets all ignored Pixels in data to NaN. List of pixels is from the
    parameter file. [(row,col), (row,col), ...]
    Args:
        data: np.array in shape (nframes, column_size, nreps, row_size)
        bad_pixels: list of tuples (row,col)
    
    Returns:
        np.array in shape (nframes, column_size, nreps, row_size)
    '''
    if np.ndim(data) != 4:
        _logger.error('Data has wrong dimensions')
        return None
    _logger.info('Excluding bad pixels')
    bad_pixel_mask = np.zeros(data.shape, dtype=bool)
    for index in bad_pixels:
        col = index[1]
        row = index[0]
        bad_pixel_mask[:,row,:,col] = True
    data[bad_pixel_mask] = np.nan
    _logger.info(f'Excluded {len(bad_pixels)} pixels')
    return data

def correct_common_mode(data):
    '''
    Calculates the median of euch row in data, and substracts it from 
    the row.
    Correction is done inline to save memory.
    Args:
        np.array in shape (nframes, column_size, nreps, row_size)
    '''
    _logger.info('Starting common mode correction.')  
    # Iterate over the nframes dimension
    for i in range(data.shape[0]):
        # Calculate the median for one frame
        median_common = np.nanmedian(data[i], axis=2, keepdims=True)
        # Subtract the median from the frame in-place
        data[i] -= median_common
    _logger.info('Data is corrected for common mode.')

def calc_event_map(avg_over_nreps, noise_fitted, thres_event):
    _logger.info('Finding events')
    threshold_map = noise_fitted * thres_event
    events = avg_over_nreps > threshold_map[np.newaxis,:,:]
    signals = avg_over_nreps[events]
    indices = np.transpose(np.where(events))
    _logger.info(f'{signals.shape[0]} events found')
    event_array = np.concatenate(
        (indices, signals[:,np.newaxis]),
          axis = 1
        )
    event_map = np.zeros((64,64), dtype = object)
    event_map.fill([])
    for entry in event_array:
        row = int(entry[1])
        column = int(entry[2])
        signal = entry[3]
        event_map[row][column] = np.append(
            event_map[row][column], signal
            )
    return event_map

def get_sum_of_event_signals(event_map, row_size, column_size):
    sum_of_events = np.zeros((row_size,column_size))
    for row in range(row_size):
        for column in range(column_size):
            sum_of_events[row][column] = sum(event_map[row][column])
    return sum_of_events

def get_sum_of_event_counts(event_map, row_size, column_size):
    sum_of_events = np.zeros((row_size,column_size))
    for row in range(row_size):
        for column in range(column_size):
            sum_of_events[row][column] = len(event_map[row][column])
    return sum_of_events

def get_gain_fit(event_map, row_size, column_size, min_signals):
    mean = np.full((row_size, column_size), np.nan)
    sigma = np.full((row_size, column_size), np.nan)
    mean_error = np.full((row_size, column_size), np.nan)
    sigma_error = np.full((row_size, column_size), np.nan)

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
            if len(signals) >= min_signals:
                params = fit_hist(signals)
                mean[i,j] = params[0]
                sigma[i,j] = params[1]
                mean_error[i,j] = params[2]
                sigma_error[i,j] = params[3]
            else:
                count_too_few += 1
    _logger.info(f'{count_too_few} pixels have less than {min_signals} signals')
    return mean, sigma, mean_error, sigma_error