TODO:
Prio:
- make roan_steps work; thre step classes should work. and write one function
    that just calculates all
- bad slopes should be ignored in the fitting in the offnoi step
- add infos to plots
- documentation of functions
- just load the prm dictionary in the roan_steps classes, dont define all those class variables.

-upload whole package to github
-check if nproanPackage needs to be uploaded
-test it with collagues
-exceptions und logging: farben für logging
-wiki updaten

-add type hints and write nice documentation:
 analysis, analysis_funcs and display modules are done
-replace all np.nanmean and np.nanmedian with its parallel versions.
-add new parallel functions

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