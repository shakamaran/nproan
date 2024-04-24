
TODO:
- add more documentation
- write wiki
- write functions for fitting histograms using minuit
- implement logging to file and exception stuff
- implement nreps_eval (parameter for offnoi and filter):
    [0,10,2] should evaluate nreps [0,2,4,6,8,10] -> [min,max,step]
- implement bad slopes (for offnoi and filter)
    and provide output for it

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

- functions that return values should be named get_something(), this is 
    preferred
- data should not be stored in the class
- data that is loaded in the filter or gain step is stored in a class variable
    and deleted after use