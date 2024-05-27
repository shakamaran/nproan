import numpy as np
from numba import njit, prange

from . import logger

_logger = logger.Logger(__name__, 'info').get_logger()

'''
Add parallized versions here.
Add nanmedian and nanmean for now.
Test if its good to always use parallel functions.
Write them only for keepdims=false and use np.newaxis if neccesary.
move to analysis funcs when they work.
'''

def nanmedian_parallel(data: np.ndarray, 
                       axis: int, 
                       keepdims: bool = False) -> np.ndarray:
    '''
    The equivalent to np.nanmedian(data, axis=axis, keepdims=keepdims).
    Args:
        data: np.array
        axis: int
        keepdims: bool
    Returns:
        np.array
    '''
    if data.ndim == 2:
        if axis == 0:
            if keepdims:
                return nanmedian_2d_axis0(data)[np.newaxis,:]
            else:
                return nanmedian_2d_axis0(data)
        elif axis == 1:
            if keepdims:
                return nanmedian_2d_axis1(data)[:,np.newaxis]
            else:
                return nanmedian_2d_axis1(data)
    elif data.ndim == 3:
        if axis == 0:
            if keepdims:
                return nanmedian_3d_axis0(data)[np.newaxis,:,:]
            else:
                return nanmedian_3d_axis0(data)
        elif axis == 1:
            if keepdims:
                return nanmedian_3d_axis1(data)[:,np.newaxis,:]
            else:
                return nanmedian_3d_axis1(data)
        elif axis == 2:
            if keepdims:
                return nanmedian_3d_axis2(data)[:,:,np.newaxis]
            else:
                return nanmedian_3d_axis2(data)
    elif data.ndim == 4:
        if axis == 0:
            if keepdims:
                return parallel_nanmedian_4d_axis0(data)[np.newaxis,:,:,:]
            else:
                return parallel_nanmedian_4d_axis0(data)
        elif axis == 1:
            if keepdims:
                return parallel_nanmedian_4d_axis1(data)[:,np.newaxis,:,:]
            else:
                return parallel_nanmedian_4d_axis1(data)
        elif axis == 2:
            if keepdims:
                return parallel_nanmedian_4d_axis2(data)[:,:,np.newaxis,:]
            else:
                return parallel_nanmedian_4d_axis2(data)
        elif axis == 3:
            if keepdims:
                return parallel_nanmedian_4d_axis3(data)[:,:,:,np.newaxis]
            else:
                return parallel_nanmedian_4d_axis3(data)
    else:
        _logger.error('Data has wrong dimensions')
        return

@njit(parallel=True)
def parallel_nanmedian_4d_axis0(data: np.ndarray) -> np.ndarray:
    '''
    The equivalent to np.nanmedian(data, axis=0, keepdims=False).
    Args:
        data: 4D np.array 
    Returns:
        3D np.array
    '''
    if data.ndim != 4:
        _logger.error('Input data is not a 4D array.')
        raise ValueError('Input data is not a 4D array.')
    axis_1_size = data.shape[1]
    axis_2_size = data.shape[2]
    axis_3_size = data.shape[3]
    output = np.zeros((axis_1_size,axis_2_size, axis_3_size))
    for i in prange(axis_1_size):
        for j in prange(axis_2_size):
            for k in prange(axis_3_size):
                median = np.nanmedian(data[:,i,j,k])
                output[i,j,k] = median
    return output

@njit(parallel=True)
def parallel_nanmedian_4d_axis1(data: np.ndarray) -> np.ndarray:
    '''
    The equivalent to np.nanmedian(data, axis=1, keepdims=False).
    Args:
        data: 4D np.array
    Returns:
        3D np.array
    '''
    if data.ndim != 4:
        _logger.error('Input data is not a 4D array.')
        raise ValueError('Input data is not a 4D array.')
    axis_0_size = data.shape[0]
    axis_2_size = data.shape[2]
    axis_3_size = data.shape[3]
    output = np.zeros((axis_0_size,axis_2_size, axis_3_size))
    for i in prange(axis_0_size):
        for j in prange(axis_2_size):
            for k in prange(axis_3_size):
                median = np.nanmedian(data[i,:,j,k])
                output[i,j,k] = median
    return output

@njit(parallel=True)
def parallel_nanmedian_4d_axis2(data: np.ndarray) -> np.ndarray:
    '''
    The equivalent to np.nanmedian(data, axis=2, keepdims=False).
    Args:
        data: 4D np.array
    Returns:
        3D np.array
    '''
    if data.ndim != 4:
        _logger.error('Input data is not a 4D array.')
        raise ValueError('Input data is not a 4D array.')
    axis_0_size = data.shape[0]
    axis_1_size = data.shape[1]
    axis_3_size = data.shape[3]
    output = np.zeros((axis_0_size,axis_1_size, axis_3_size))
    for i in prange(axis_0_size):
        for j in prange(axis_1_size):
            for k in prange(axis_3_size):
                median = np.nanmedian(data[i,j,:,k])
                output[i,j,k] = median
    return output

@njit(parallel=True)
def parallel_nanmedian_4d_axis3(data: np.ndarray) -> np.ndarray:
    '''
    The equivalent to np.nanmedian(data, axis=3, keepdims=False).
    Args:
        data: 4D np.array
    Returns:
        3D np.array
    '''
    if data.ndim != 4:
        _logger.error('Input data is not a 4D array.')
        raise ValueError('Input data is not a 4D array.')
    axis_0_size = data.shape[0]
    axis_1_size = data.shape[1]
    axis_2_size = data.shape[2]  
    output = np.zeros((axis_0_size,axis_1_size, axis_2_size))
    for i in prange(axis_0_size):
        for j in prange(axis_1_size):
            for k in prange(axis_2_size):
                median = np.nanmedian(data[i,j,k,:])
                output[i,j,k] = median
    return output

@njit(parallel=True)
def nanmedian_3d_axis0(data: np.ndarray) -> np.ndarray:
    '''
    The equivalent to np.nanmedian(data, axis=0, keepdims=False).
    Args:
        data: 3D np.array
    Returns:
        2D np.array
    '''
    if data.ndim != 3:
        _logger.error('Input data is not a 3D array.')
        raise ValueError('Input data is not a 3D array.')
    axis_1_size = data.shape[1]
    axis_2_size = data.shape[2]
    output = np.zeros((axis_1_size, axis_2_size))
    for i in prange(axis_1_size):
        for j in prange(axis_2_size):
            median = np.nanmedian(data[:,i,j])
            output[i,j] = median
    return output

@njit(parallel=True)
def nanmedian_3d_axis1(data: np.ndarray) -> np.ndarray:
    '''
    The equivalent to np.nanmedian(data, axis=1, keepdims=False).
    Args:
        data: 3D np.array
    Returns:
        2D np.array
    '''
    if data.ndim != 3:
        _logger.error('Input data is not a 3D array.')
        raise ValueError('Input data is not a 3D array.')
    axis_0_size = data.shape[0]
    axis_2_size = data.shape[2]
    output = np.zeros((axis_0_size, axis_2_size))
    for i in prange(axis_0_size):
        for j in prange(axis_2_size):
            median = np.nanmedian(data[i,:,j])
            output[i,j] = median
    return output

@njit(parallel=True)
def nanmedian_3d_axis2(data: np.ndarray) -> np.ndarray:
    '''
    The equivalent to np.nanmedian(data, axis=2, keepdims=False).
    Args:
        data: 3D np.array
    Returns:
        2D np.array
    '''
    if data.ndim != 3:
        _logger.error('Input data is not a 3D array.')
        raise ValueError('Input data is not a 3D array.')
    axis_0_size = data.shape[0]
    axis_1_size = data.shape[1]
    output = np.zeros((axis_0_size, axis_1_size))
    for i in prange(axis_0_size):
        for j in prange(axis_1_size):
            median = np.nanmedian(data[i,j,:])
            output[i,j] = median
    return output

@njit(parallel=True)
def nanmedian_2d_axis0(data: np.ndarray) -> np.ndarray:
    '''
    The equivalent to np.nanmedian(data, axis=0, keepdims=False).
    Args:
        data: 2D np.array
    Returns:
        1D np.array
    '''
    if data.ndim != 2:
        _logger.error('Input data is not a 2D array.')
        raise ValueError('Input data is not a 2D array.')
    axis_1_size = data.shape[1]
    output = np.zeros(axis_1_size)
    for i in prange(axis_1_size):
        median = np.nanmedian(data[:,i])
        output[i] = median
    return output

@njit(parallel=True)
def nanmedian_2d_axis1(data: np.ndarray) -> np.ndarray:

    '''
    The equivalent to np.nanmedian(data, axis=1, keepdims=False).
    Args:
        data: 2D np.array
    Returns:
        1D np.array
    '''
    if data.ndim != 2:
        _logger.error('Input data is not a 2D array.')
        raise ValueError('Input data is not a 2D array.')
    axis_0_size = data.shape[0]
    output = np.zeros(axis_0_size)
    for i in prange(axis_0_size):
        median = np.nanmedian(data[i,:])
        output[i] = median
    return output

def nanmean_parallel(data: np.ndarray, 
                       axis: int, 
                       keepdims: bool = False) -> np.ndarray:
    '''
    The equivalent to np.nanmean(data, axis=axis, keepdims=keepdims).
    Args:
        data: np.array
        axis: int
        keepdims: bool
    Returns:
        np.array
    '''
    if data.ndim == 2:
        if axis == 0:
            if keepdims:
                return nanmean_2d_axis0(data)[np.newaxis,:]
            else:
                return nanmean_2d_axis0(data)
        elif axis == 1:
            if keepdims:
                return nanmean_2d_axis1(data)[:,np.newaxis]
            else:
                return nanmean_2d_axis1(data)
    elif data.ndim == 3:
        if axis == 0:
            if keepdims:
                return nanmean_3d_axis0(data)[np.newaxis,:,:]
            else:
                return nanmean_3d_axis0(data)
        elif axis == 1:
            if keepdims:
                return nanmean_3d_axis1(data)[:,np.newaxis,:]
            else:
                return nanmean_3d_axis1(data)
        elif axis == 2:
            if keepdims:
                return nanmean_3d_axis2(data)[:,:,np.newaxis]
            else:
                return nanmean_3d_axis2(data)
    elif data.ndim == 4:
        if axis == 0:
            if keepdims:
                return parallel_nanmean_4d_axis0(data)[np.newaxis,:,:,:]
            else:
                return parallel_nanmean_4d_axis0(data)
        elif axis == 1:
            if keepdims:
                return parallel_nanmean_4d_axis1(data)[:,np.newaxis,:,:]
            else:
                return parallel_nanmean_4d_axis1(data)
        elif axis == 2:
            if keepdims:
                return parallel_nanmean_4d_axis2(data)[:,:,np.newaxis,:]
            else:
                return parallel_nanmean_4d_axis2(data)
        elif axis == 3:
            if keepdims:
                return parallel_nanmean_4d_axis3(data)[:,:,:,np.newaxis]
            else:
                return parallel_nanmean_4d_axis3(data)
    else:
        _logger.error('Data has wrong dimensions')
        return

@njit(parallel=True)
def parallel_nanmean_4d_axis0(data: np.ndarray) -> np.ndarray:
    '''
    The equivalent to np.nanmean(data, axis=0, keepdims=False).
    Args:
        data: 4D np.array 
    Returns:
        3D np.array
    '''
    if data.ndim != 4:
        _logger.error('Input data is not a 4D array.')
        raise ValueError('Input data is not a 4D array.')
    axis_1_size = data.shape[1]
    axis_2_size = data.shape[2]
    axis_3_size = data.shape[3]
    output = np.zeros((axis_1_size,axis_2_size, axis_3_size))
    for i in prange(axis_1_size):
        for j in prange(axis_2_size):
            for k in prange(axis_3_size):
                median = np.nanmean(data[:,i,j,k])
                output[i,j,k] = median
    return output

@njit(parallel=True)
def parallel_nanmean_4d_axis1(data: np.ndarray) -> np.ndarray:
    '''
    The equivalent to np.nanmean(data, axis=1, keepdims=False).
    Args:
        data: 4D np.array
    Returns:
        3D np.array
    '''
    if data.ndim != 4:
        _logger.error('Input data is not a 4D array.')
        raise ValueError('Input data is not a 4D array.')
    axis_0_size = data.shape[0]
    axis_2_size = data.shape[2]
    axis_3_size = data.shape[3]
    output = np.zeros((axis_0_size,axis_2_size, axis_3_size))
    for i in prange(axis_0_size):
        for j in prange(axis_2_size):
            for k in prange(axis_3_size):
                median = np.nanmean(data[i,:,j,k])
                output[i,j,k] = median
    return output

@njit(parallel=True)
def parallel_nanmean_4d_axis2(data: np.ndarray) -> np.ndarray:
    '''
    The equivalent to np.nanmean(data, axis=2, keepdims=False).
    Args:
        data: 4D np.array
    Returns:
        3D np.array
    '''
    if data.ndim != 4:
        _logger.error('Input data is not a 4D array.')
        raise ValueError('Input data is not a 4D array.')
    axis_0_size = data.shape[0]
    axis_1_size = data.shape[1]
    axis_3_size = data.shape[3]
    output = np.zeros((axis_0_size,axis_1_size, axis_3_size))
    for i in prange(axis_0_size):
        for j in prange(axis_1_size):
            for k in prange(axis_3_size):
                median = np.nanmean(data[i,j,:,k])
                output[i,j,k] = median
    return output

@njit(parallel=True)
def parallel_nanmean_4d_axis3(data: np.ndarray) -> np.ndarray:
    '''
    The equivalent to np.nanmean(data, axis=3, keepdims=False).
    Args:
        data: 4D np.array
    Returns:
        3D np.array
    '''
    if data.ndim != 4:
        _logger.error('Input data is not a 4D array.')
        raise ValueError('Input data is not a 4D array.')
    axis_0_size = data.shape[0]
    axis_1_size = data.shape[1]
    axis_2_size = data.shape[2]  
    output = np.zeros((axis_0_size,axis_1_size, axis_2_size))
    for i in prange(axis_0_size):
        for j in prange(axis_1_size):
            for k in prange(axis_2_size):
                median = np.nanmean(data[i,j,k,:])
                output[i,j,k] = median
    return output

@njit(parallel=True)
def nanmean_3d_axis0(data: np.ndarray) -> np.ndarray:
    '''
    The equivalent to np.nanmean(data, axis=0, keepdims=False).
    Args:
        data: 3D np.array
    Returns:
        2D np.array
    '''
    if data.ndim != 3:
        _logger.error('Input data is not a 3D array.')
        raise ValueError('Input data is not a 3D array.')
    axis_1_size = data.shape[1]
    axis_2_size = data.shape[2]
    output = np.zeros((axis_1_size, axis_2_size))
    for i in prange(axis_1_size):
        for j in prange(axis_2_size):
            median = np.nanmean(data[:,i,j])
            output[i,j] = median
    return output

@njit(parallel=True)
def nanmean_3d_axis1(data: np.ndarray) -> np.ndarray:
    '''
    The equivalent to np.nanmean(data, axis=1, keepdims=False).
    Args:
        data: 3D np.array
    Returns:
        2D np.array
    '''
    if data.ndim != 3:
        _logger.error('Input data is not a 3D array.')
        raise ValueError('Input data is not a 3D array.')
    axis_0_size = data.shape[0]
    axis_2_size = data.shape[2]
    output = np.zeros((axis_0_size, axis_2_size))
    for i in prange(axis_0_size):
        for j in prange(axis_2_size):
            median = np.nanmean(data[i,:,j])
            output[i,j] = median
    return output

@njit(parallel=True)
def nanmean_3d_axis2(data: np.ndarray) -> np.ndarray:
    '''
    The equivalent to np.nanmean(data, axis=2, keepdims=False).
    Args:
        data: 3D np.array
    Returns:
        2D np.array
    '''
    if data.ndim != 3:
        _logger.error('Input data is not a 3D array.')
        raise ValueError('Input data is not a 3D array.')
    axis_0_size = data.shape[0]
    axis_1_size = data.shape[1]
    output = np.zeros((axis_0_size, axis_1_size))
    for i in prange(axis_0_size):
        for j in prange(axis_1_size):
            median = np.nanmean(data[i,j,:])
            output[i,j] = median
    return output

@njit(parallel=True)
def nanmean_2d_axis0(data: np.ndarray) -> np.ndarray:
    '''
    The equivalent to np.nanmean(data, axis=0, keepdims=False).
    Args:
        data: 2D np.array
    Returns:
        1D np.array
    '''
    if data.ndim != 2:
        _logger.error('Input data is not a 2D array.')
        raise ValueError('Input data is not a 2D array.')
    axis_1_size = data.shape[1]
    output = np.zeros(axis_1_size)
    for i in prange(axis_1_size):
        median = np.nanmean(data[:,i])
        output[i] = median
    return output

@njit(parallel=True)
def nanmean_2d_axis1(data: np.ndarray) -> np.ndarray:
    
    '''
    The equivalent to np.nanmean(data, axis=1, keepdims=False).
    Args:
        data: 2D np.array
    Returns:
        1D np.array
    '''
    if data.ndim != 2:
        _logger.error('Input data is not a 2D array.')
        raise ValueError('Input data is not a 2D array.')
    axis_0_size = data.shape[0]
    output = np.zeros(axis_0_size)
    for i in prange(axis_0_size):
        median = np.nanmean(data[i,:])
        output[i] = median
    return output