import os
import fnmatch
import json

class Params:
    '''
    To change/add parameters, edit/add them here.
    Also change the load() function in the respective class.
    '''
    common_params = {
        'common_results_dir': "",           #str
        'common_column_size': 64,           #int
        'common_row_size': 64,              #int
        'common_key_ints': 3,               #int
        'common_bad_pixels': []             #list of tuples
    }
    offnoi_params = {
        'offnoi_bin_file': "",              #str
        'offnoi_nreps': 200,                 #int
        'offnoi_nframes': 100,                #int
        'offnoi_nreps_eval': [0,-1,1],        #list of ints
        'offnoi_comm_mode': True,             #bool
        'offnoi_thres_mips': 1000,            #float
        'offnoi_thres_bad_frames': 5,          #float
        'offnoi_thres_bad_slopes': 5          #float
    }
    filter_params = {
        'filter_bin_file': "",            #str
        'filter_nreps': 200,               #int
        'filter_nframes': 100,              #int
        'filter_nreps_eval': [0,-1,1],           #list of ints
        'filter_comm_mode': True,           #bool
        'filter_thres_mips': 1000,          #float
        'filter_thres_event': 5,            #float
        'filter_use_fitted_offset': False,  #bool
        'filter_thres_bad_frames': 5,        #float
        'filter_thres_bad_slopes': 5        #float
    }
    gain_params = {
        'gain_min_signals': 5               #int
    }

    params_types = {
        'common_results_dir': str,
        'common_column_size': int,
        'common_row_size': int,
        'common_key_ints': int,
        'common_bad_pixels': list,          #actually, list of tuples

        'offnoi_bin_file': str,
        'offnoi_nreps': int,
        'offnoi_nframes': int,
        'offnoi_nreps_eval': list,
        'offnoi_comm_mode': bool,
        'offnoi_thres_mips': (int, float),
        'offnoi_thres_bad_frames': (int, float),
        'offnoi_thres_bad_slopes': (int, float),

        'filter_bin_file': str,
        'filter_nreps': int,
        'filter_nframes': int,
        'filter_nreps_eval': list,
        'filter_comm_mode': bool,
        'filter_thres_mips': (int, float),
        'filter_thres_event': (int, float),
        'filter_use_fitted_offset': bool,
        'filter_thres_bad_frames': (int, float),
        'filter_thres_bad_slopes': (int, float),

        'gain_min_signals': int
    }

    #required parameters, where there is no default value
    #file cannot be loaded if these are missing
    required_params = [
        'offnoi_bin_file',
        'offnoi_nreps',
        'filter_bin_file',
        'filter_nreps'
    ]

    def __init__(self, json_path=None):
        self.default_dict = {**self.common_params,
                             **self.offnoi_params,
                             **self.filter_params,
                             **self.gain_params}
        self.inp_dict = None
        self.param_dict = None
        if json_path is not None:
            self.update(json_path)
        else:
            print('No parameter file provided.')
            print('Run save_default_file() to save a default parameter file.')
        self.check_types()

    def update(self, json_path):
        try:
            with open(json_path) as f:
                self.inp_dict = json.load(f)
        except:
            print('Error loading the parameter file.')
            self.save_default_file()
            print('A default parameter file has been saved to the current directory.')
            self.param_dict = None
            return
        self.param_dict = self.default_dict.copy()
        #check consistency of the input dict with the default dict
        for key,value in self.inp_dict.items():
            if key not in self.default_dict:
                print(f"{key} is not a valid parameter.")
            else:
                self.param_dict[key] = value
        #check for missing parameters, using default if not required
        #if parameter has no default, set param_dict to None
        for key,value in self.param_dict.items():
            if value is None:
                if key in self.required_params:
                    print(f"{key} is missing in the file.")
                    print('Please provide a complete parameter file')
                    print('Run .info() to see the required parameters.')
                    self.param_dict = None
                    break
                else:
                    print(f"{key} is missing. Using default: {self.default_dict[key]}")

    def check_types(self):
        for key,value in self.param_dict.items():
            if key not in self.params_types:
                raise TypeError(f"There is no type defined for {key}.")
            else:
                expected_type = self.params_types[key]
                if not isinstance(value, expected_type):
                    raise TypeError(f"Expected {key} to be of type {expected_type}.")
                
    def get_dict(self):
        return self.param_dict

    def print_contents(self):
        for key, value in self.param_dict.items():
            print(f"{key}: {value}")

    def info(self):
        print('The following parameters must be provided:')
        print('--common parameters:')
        for key in self.common_params.keys():
            print(key)
        print('--offnoi parameters:')
        for key in self.offnoi_params.keys():
            print(key)
        print('--filter parameters:')
        for key in self.filter_params.keys():
            print(key)
        print('--gain parameters:')
        for key in self.gain_params.keys():
            print(key)

    def save_default_file(self, path=None):
        #if no path is provided, save to the current directory
        print(f'path: {path}')
        if path is None:
            path = os.path.join(os.getcwd(), 'default_params.json')
        else:
            path = os.path.join(path, 'default_params.json')
        with open(path, 'w') as f:
            json.dump(self.default_dict, f, indent=4)
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.param_dict, f, indent=4)

    def get_json_file_name_in_folder(self, folder_path):
        count = 0
        json_file =''
        for file in os.listdir(folder_path):
            if fnmatch.fnmatch(file, '*.json'):
                json_file = os.path.join(folder_path, file)
                count += 1
        if count == 1:
            return json_file
        else:
            return False
            
    def same_common_params(self, folder_path):
        json_file = self.get_json_file_name_in_folder(folder_path)
        print(f'json_file: {json_file}')
        if json_file:
            with open(json_file) as f:
                dict = json.load(f)
            for key in self.common_params.keys():
                if self.param_dict[key] != dict[key]:
                    return False
            return True
        else:
            print('No json file found in the folder')
            return False
        
    def same_offnoi_params(self, folder_path):
        json_file = self.get_json_file_name_in_folder(folder_path)
        if json_file:
            with open(json_file) as f:
                dict = json.load(f)
            for key in self.offnoi_params.keys():
                if self.param_dict[key] != dict[key]:
                    return False
            return True
        else:
            print('No json file found in the folder')
            return False
    
    def same_filter_params(self, folder_path):
        json_file = self.get_json_file_name_in_folder(folder_path)
        if json_file:
            with open(json_file) as f:
                dict = json.load(f)
            for key in self.filter_params.keys():
                if self.param_dict[key] != dict[key]:
                    return False
            return True
        else:
            print('No json file found in the folder')
            return False
        
    def same_gain_params(self, folder_path):
        json_file = self.get_json_file_name_in_folder(folder_path)
        if json_file:
            with open(json_file) as f:
                dict = json.load(f)
            for key in self.gain_params.keys():
                if self.param_dict[key] != dict[key]:
                    return False
            return True
        else:
            print('No json file found in the folder')
            return False