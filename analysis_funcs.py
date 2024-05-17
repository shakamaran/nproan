import gc
import os

import numpy as np

import logger

_logger = logger.Logger(__name__, 'info').get_logger()

def get_avg_over_frames(data):
    '''
    Calculates the average over the frames in data.
    Args:
        np.array in shape (nframes, column_size, nreps, row_size)
    
    Returns:
        np.array in shape (column_size, nreps, row_size)
    '''
    if np.ndim(data) != 4:
        _logger.error('Data has wrong dimensions')
        return None
    return np.nanmean(data, axis = 0)

def get_avg_over_nreps(data):
   '''
   Calculates the average over the nreps in data.

   Args:
       np.array in shape (nframes, column_size, nreps, row_size)

   Returns:
       np.array in shape (nframes, column_size, row_size)
   '''
   if np.ndim(data) != 4:
       _logger.error('Data has wrong dimensions')
       return None
   return np.nanmean(data, axis = 2)

def get_avg_over_frames_and_nreps(data):
    '''
    Calculates the average over the frames and nreps in data.
    Args:
        np.array in shape (nframes, column_size, nreps, row_size)
    Returns:
        np.array in shape (column_size, row_size)
    '''
    if np.ndim(data) != 4:
        _logger.error('Data has wrong dimensions')
        return None
    return np.nanmean(data, axis = (0,2))

def set_pixels_to_nan(data, indices):
    '''
    sets all pixels in indices in the data array to nan and returns a copy.
    
    Args:
        np.array in shape (nframes, column_size, row_size)
        list of tuples [(frame, row, column), (frame, row, column), ...]
    '''
    if np.ndim(data) != 3:
        _logger.error('Data has wrong dimensions')
        return None
    if np.ndim(indices) != 2:
        _logger.error('Pixel positions have wrong dimensions')
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
    '''
    Looks for .npy arrays in folder and returns them all as a
    dictionary with numpy arrays as values

    Args:
        folder: folder path

    Returns:
        dictionary of np.arrays
    '''
    # Get a list of all .npy files in the folder
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    if len(files) == 0:
        print(f'No .npy files found in folder {folder}')
        return None
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

def get_array_from_file(folder, filename):
    return np.load(os.path.join(folder, filename), allow_pickle=True)