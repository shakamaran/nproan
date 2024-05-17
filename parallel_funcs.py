import numpy as np
import numba as nb

import logger

_logger = logger.Logger(__name__, 'info').get_logger()

'''
Add parallized versions here.
Add nanmedian and nanmean for now.
Test if its good to always use parallel functions.
Write them only for keepdims=false and use np.newaxis if neccesary.
move to analysis funcs when they work.
'''
@nb.jit(nopython=True, parallel=True)
def nanmedian_nb(data, axis, keepdims=False):
    #keepdims=True
    #assume a 4d Array!
    if data.ndim != 4:
        print('Wrong dimensions')
        return 0
    frames = data.shape[0]
    rows = data.shape[1]
    nreps = data.shape[2]
    cols = data.shape[3]

    if axis == 0 and keepdims:
        output = np.zeros((1, rows, nreps, cols))
        for row in nb.prange(rows):
            for nrep in nb.prange(nreps):
                for col in nb.prange(cols):
                    median = np.nanmedian(data[:,row,nrep,col])
                    output[0,row,nrep,col] = median
        return output
    
    if axis == 1 and keepdims:
        output = np.zeros((frames, 1, nreps, cols))
        for frame in nb.prange(frames):
            for nrep in nb.prange(nreps):
                for col in nb.prange(cols):
                    median = np.nanmedian(data[frame,:,nrep,col])
                    output[frame,0,nrep,col] = median
        return output
    
    if axis == 2 and keepdims:
        output = np.zeros((frames, rows, 1, cols))
        for frame in nb.prange(frames):
            for row in nb.prange(rows):
                for col in nb.prange(cols):
                    median = np.nanmedian(data[frame,row,:,col])
                    output[frame,row,0,col] = median
        return output
    
    if axis == 3 and keepdims:
        output = np.zeros((frames, rows, nreps, 1))
        for frame in nb.prange(frames):
            for row in nb.prange(rows):
                for nrep in nb.prange(nreps):
                    median = np.nanmedian(data[frame,row,nrep,:])
                    output[frame,row,nrep,0] = median
        return output
    
    if axis == 0 and not keepdims:
        output = np.zeros((rows, nreps, cols))
        for row in nb.prange(rows):
            for nrep in nb.prange(nreps):
                for col in nb.prange(cols):
                    median = np.nanmedian(data[:,row,nrep,col])
                    output[row,nrep,col] = median
        return output
    
    if axis == 1 and not keepdims:
        output = np.zeros((frames, nreps, cols))
        for frame in nb.prange(frames):
            for nrep in nb.prange(nreps):
                for col in nb.prange(cols):
                    median = np.nanmedian(data[frame,:,nrep,col])
                    output[frame,nrep,col] = median
        return output
    
    if axis == 2 and not keepdims:
        output = np.zeros((frames, rows, cols))
        for frame in nb.prange(frames):
            for row in nb.prange(rows):
                for col in nb.prange(cols):
                    median = np.nanmedian(data[frame,row,:,col])
                    output[frame,row,col] = median
        return output
    
    if axis == 3 and not keepdims:
        output = np.zeros((frames, rows, nreps))
        for frame in nb.prange(frames):
            for row in nb.prange(rows):
                for nrep in nb.prange(nreps):
                    median = np.nanmedian(data[frame,row,nrep,:])
                    output[frame,row,nrep] = median
        return output

@nb.jit(nopython=True, parallel=True)
def nanmean_nb(data, axis, keepdims=False):
    #keepdims=True
    #assume a 4d Array!
    if data.ndim != 4:
        print('Wrong dimensions')
        return 0
    frames = data.shape[0]
    rows = data.shape[1]
    nreps = data.shape[2]
    cols = data.shape[3]

    if axis == 0 and keepdims:
        output = np.zeros((1, rows, nreps, cols))
        for row in nb.prange(rows):
            for nrep in nb.prange(nreps):
                for col in nb.prange(cols):
                    median = np.nanmean(data[:,row,nrep,col])
                    output[0,row,nrep,col] = median
        return output
    
    if axis == 1 and keepdims:
        output = np.zeros((frames, 1, nreps, cols))
        for frame in nb.prange(frames):
            for nrep in nb.prange(nreps):
                for col in nb.prange(cols):
                    median = np.nanmean(data[frame,:,nrep,col])
                    output[frame,0,nrep,col] = median
        return output
    
    if axis == 2 and keepdims:
        output = np.zeros((frames, rows, 1, cols))
        for frame in nb.prange(frames):
            for row in nb.prange(rows):
                for col in nb.prange(cols):
                    median = np.nanmean(data[frame,row,:,col])
                    output[frame,row,0,col] = median
        return output
    
    if axis == 3 and keepdims:
        output = np.zeros((frames, rows, nreps, 1))
        for frame in nb.prange(frames):
            for row in nb.prange(rows):
                for nrep in nb.prange(nreps):
                    median = np.nanmean(data[frame,row,nrep,:])
                    output[frame,row,nrep,0] = median
        return output
    
    if axis == 0 and not keepdims:
        output = np.zeros((rows, nreps, cols))
        for row in nb.prange(rows):
            for nrep in nb.prange(nreps):
                for col in nb.prange(cols):
                    median = np.nanmean(data[:,row,nrep,col])
                    output[row,nrep,col] = median
        return output
    
    if axis == 1 and not keepdims:
        output = np.zeros((frames, nreps, cols))
        for frame in nb.prange(frames):
            for nrep in nb.prange(nreps):
                for col in nb.prange(cols):
                    median = np.nanmean(data[frame,:,nrep,col])
                    output[frame,nrep,col] = median
        return output
    
    if axis == 2 and not keepdims:
        output = np.zeros((frames, rows, cols))
        for frame in nb.prange(frames):
            for row in nb.prange(rows):
                for col in nb.prange(cols):
                    median = np.nanmean(data[frame,row,:,col])
                    output[frame,row,col] = median
        return output
    
    if axis == 3 and not keepdims:
        output = np.zeros((frames, rows, nreps))
        for frame in nb.prange(frames):
            for row in nb.prange(rows):
                for nrep in nb.prange(nreps):
                    median = np.nanmean(data[frame,row,nrep,:])
                    output[frame,row,nrep] = median
        return output