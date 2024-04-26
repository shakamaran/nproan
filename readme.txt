TODO:

- test bad slopes and nreps eval

- start analysing some files in notebooks and add functions to common.py
    check the fitting while youre at it

- remove output of pictures
- add documentation to functions
- implement logging to file and exception stuff

- write wiki and an example file for Analysis
- write functions for fitting histograms using minuit (?)



My current philosophy for doing stuff:
- np roan should calculate stuff for now, functions for drawing/saving can
    be put in common.py
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
- data should not be stored in the class
- data that is loaded in the filter or gain step is stored in a class variable
    and deleted after use