TODO:
Prio:
- bad slopes should be ignored in the fitting in the offnoi step
- add infos to plots

-upload whole package to github
-check if nproanPackage needs to be uploaded
-test it with collagues
-exceptions und logging: farben für logging
-wiki updaten

#TODO: Reorganize modules/inheritance:

Step module:
roan_steps.py

Is initialized with the parameter file path.
Parameters are class variables.
There should be no methods other than variants of a calculate() method.

The calculate() method performs "standard" data manipulation and stores data 
in files. all steps in the calculate() method should be possible in the
jupyter notebook!
So all methods must be independent from the class!
roan_steps.py should do everything the three steps do now.

All methods, that manipulate data should be contained in seperate modules:
-logger.py (as is)
-params.py (as is)
-display.py (split common)
-analysis.py (split common, eg: bad slopes)
-analysis_funcs.py(split common, functions used in analysis.py eg: np.nanmedian)
-fitting.py (split common)

Implement logging like here:
https://stackoverflow.com/questions/15727420/using-logging-in-multiple-modules


COL vs ROW Convention:

In ROOT its (col, row), but:
data is represented as (frame,row,nreps,col), so i will use (row, col) here
Size if data is (frames,column_size,nreps, row_size)


My current philosophy for doing stuff:
- Parameters from the parameter file are stored in the class variables
- Parameters can be boolean or int. for thresholds 0 means no calculation is performed.
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
- functions that return values should be named get_something(), its preferred To
    return a value. 
- provide some functions that do data manipulation inplace
- data that is loaded from file in the filter or gain step is stored in a class 
    variable



Parallelization using numba:

@nb.jit(nopython=True, parallel=True)
def nanmedian_nb(data):
    #axis = 3
    frames = data.shape[0]
    rows = data.shape[1]
    nreps = data.shape[2]
    output = np.zeros((frames, rows, nreps, 1))
    for frame in nb.prange(frames):
        for row in nb.prange(rows):
            for nrep in nb.prange(nreps):
                median = np.nanmedian(data[frame,row,nrep,:])
                output[frame,row,nrep,0] = median
    return output

@nb.jit(nopython=True)
def common_nb(data):
    return data - nanmedian_nb(data)