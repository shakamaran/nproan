import os
import json

class Params:

    common_params = {
        'common_column_size': 64,           #int
        'common_row_size': 64,              #int
        'common_key_ints': 3,               #int
        'common_bad_pixels': None           #list of tuples (column,row)
    }
    signal_params = {
        'signal_bin_file': None,            #str
        'signal_nreps': None,               #int
        'signal_nframes': 100,              #int
        'signal_comm_mode': True,           #bool
        'signal_thres_mips': None,          #int
        'signal_thres_event': 5,            #int
        'signal_use_fitted_offset': True    #bool
    }
    dark_params = {
        'dark_bin_file': None,              #str
        'dark_nreps': None,                 #int
        'dark_nframes': 100,                #int
        'dark_comm_mode': True,             #bool
        'dark_thres_mips': None             #int
    }
    #required parameters, where there is no default value
    #file cannot be loaded if these are missing
    required_params = [
        'signal_bin_file',
        'signal_nreps',
        'dark_bin_file',
        'dark_nreps'
    ]


    def __init__(self, json_path):
        self.default_dict = {**self.common_params, **self.signal_params, **self.dark_params}
        self.update(json_path)

    def update(self, json_path):
        with open(json_path) as f:
            self.inp_dict = json.load(f)
        self.out_dict = self.default_dict.copy()
        #check consistency of the input dict with the default dict
        for key,value in self.inp_dict.items():
            if key not in self.default_dict:
                print(f"{key} is not a valid parameter.")
            else:
                self.out_dict[key] = value
        #check for missing parameters, using default if not required
        #if parameter has no default, set out_dict to None
        for key,value in self.out_dict.items():
            if value is None:
                if key in self.required_params:
                    print(f"{key} is missing in the file.")
                    print('Please provide a complete parameter file')
                    print('Run .info() to see the required parameters.')
                    self.out_dict = None
                    break
                else:
                    print(f"{key} is missing. Using default: {self.default_dict[key]}")
                
    def get_dict(self):
        return self.out_dict

    def print_contents(self):
        for key, value in self.out_dict.items():
            print(f"{key}: {value}")

    def info(self):
        print('The following parameters must be provided:')
        print('--common parameters:')
        for key in self.common_params.keys():
            print(key)
        print('--signal parameters:')
        for key in self.signal_params.keys():
            print(key)
        print('--dark parameters:')
        for key in self.dark_params.keys():
            print(key)

    def save_default_file(self, path=None):
        #if no path is provided, save to the current directory
        if path is None:
            path = os.path.join(os.getcwd(), 'default_params.json')
        with open(path, 'w') as f:
            json.dump(self.default_dict, f, indent=4)
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.out_dict, f, indent=4)