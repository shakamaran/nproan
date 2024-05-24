import numpy as np
from numba import jit, prange

from . import logger

_logger = logger.Logger(__name__, 'info').get_logger()

'''
Add parallized versions here.
Add nanmedian and nanmean for now.
Test if its good to always use parallel functions.
Write them only for keepdims=false and use np.newaxis if neccesary.
move to analysis funcs when they work.
'''
@jit(nopython=True, parallel=True)
def nanmedian_3d_axis2(data):
    '''
    The equivalent to np.nanmedian(data, axis=2, keepdims=False).

    Args:
        data: 3D np.array in shape (rows, nreps, cols)

    Returns:
        np.array in shape (rows, nreps, 1)
    '''
    #keepdims=True
    #assume a 4d Array!
    if data.ndim != 3:
        print('Wrong dimensions')
        return 0
    rows = data.shape[0]
    nreps = data.shape[1]
    
    output = np.zeros((rows, nreps))
    for row in prange(rows):
        for nrep in prange(nreps):
            median = np.nanmedian(data[row,nrep,:])
            output[row,nrep] = median
    return output