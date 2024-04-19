import os
import fnmatch
import json

class Params:
    '''
    To change the parameters, edit them here.
    Also change the load() function in the respective class.
    '''
    common_params = {
        'common_results_dir': None,         #str
        'common_column_size': 64,           #int
        'common_row_size': 64,              #int
        'common_key_ints': 3,               #int
        'common_bad_pixels': None           #list of tuples (column,row)
    }
    offnoi_params = {
        'offnoi_bin_file': None,              #str
        'offnoi_nreps': None,                 #int
        'offnoi_nframes': 100,                #int
        'offnoi_comm_mode': True,             #bool
        'offnoi_thres_mips': None,            #int
        'offnoi_thres_bad_frames': 5          #int
    }
    filter_params = {
        'filter_bin_file': None,            #str
        'filter_nreps': None,               #int
        'filter_nframes': 100,              #int
        'filter_comm_mode': True,           #bool
        'filter_thres_mips': None,          #int
        'filter_thres_event': 5,            #int
        'filter_use_fitted_offset': False,  #bool
        'filter_thres_bad_frames': 5        #int
    }
    gain_params = {
        'gain_min_signals': 5               #int
    }
    
    #required parameters, where there is no default value
    #file cannot be loaded if these are missing
    required_params = [
        'offnoi_bin_file',
        'offnoi_nreps',
        'filter_bin_file',
        'filter_nreps'
    ]


    def __init__(self, json_path):
        self.default_dict = {**self.common_params,
                             **self.offnoi_params,
                             **self.filter_params,
                             **self.gain_params}
        self.update(json_path)

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
        #if parameter has no default, set out_dict to None
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
        print('--signal parameters:')
        for key in self.signal_params.keys():
            print(key)
        print('--dark parameters:')
        for key in self.dark_params.keys():
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