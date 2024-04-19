import gc
import os
from datetime import datetime

import numpy as np
from scipy.optimize import curve_fit

import common as cm
import params as pm

class Filter(cm.Common):
    def __init__(self, prm_file, offnoi_dir):
        self.load(prm_file, offnoi_dir)
        print('Filter object created')

    def load(self, prm_file, offnoi_dir):
        self.prm = pm.Params(prm_file)
        parameters = self.prm.get_dict()
        #common parameters
        self.results_dir = parameters['common_results_dir']
        self.column_size = parameters['common_column_size']
        self.row_size = parameters['common_row_size']
        self.key_ints = parameters['common_key_ints']
        self.bad_pixels = parameters['common_bad_pixels']
        #filter parameters
        self.bin_file = parameters['filter_bin_file']
        self.nreps = parameters['filter_nreps']
        self.nframes = parameters['filter_nframes']
        self.comm_mode = parameters['filter_comm_mode']
        self.thres_mips = parameters['filter_thres_mips']
        self.thres_event = parameters['filter_thres_event']
        self.use_fitted_offset = parameters['filter_use_fitted_offset']

        #directories
        #set self.common_dir to the parent directory of offnoi_dir
        self.common_dir = os.path.dirname(offnoi_dir)
        self.step_dir = None
        
        print(f'Parameters loaded:')
        self.prm.print_contents()
        
        print('Checking parameters in offnoi directory')
        #look for a json file in the offnoi directory 
        if (not self.prm.same_common_params(offnoi_dir)) \
            or (not self.prm.same_offnoi_params(offnoi_dir)):
            print('Parameters in offnoi directory do not match')
            return
        try:
            #offset_raw is quite big. deleted after use
            self.offset_raw = cm.get_array_from_file(
                offnoi_dir, 'offset_raw.npy')
            print(self.offset_raw.shape)
            if self.offset_raw is None:
                print('Error loading offset_raw data\n')
                return
            self.offset_fitted = cm.get_array_from_file(
                offnoi_dir, 'fitted_offset.npy'
            )
            if self.offset_fitted is None:
                print('Error loading fitted_offset data\n')
                return
            self.noise_fitted = cm.get_array_from_file(
                offnoi_dir, 'fitted_noise.npy'
            )
            if self.noise_fitted is None:
                print('Error loading fitted_noise data\n')
                return
            self.offnoi_dir = offnoi_dir
            self.common_dir = os.path.dirname(offnoi_dir)
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            self.step_dir = os.path.join(
                self.common_dir, timestamp + f'_filter_{self.thres_event}_threshold'
            )
        except:
            print('Error loading offnoi data\n')
            return

    def calculate(self):
        #create the working directory for the filter step
        os.makedirs(self.step_dir, exist_ok=True)
        print(f'Created directory for filter step: {self.step_dir}')
        # and save the parameter file there
        self.prm.save(os.path.join(self.step_dir, 'parameters.json'))

        data = self.get_data()
        #omit bad pixels and mips frames
        if self.bad_pixels is not None:
            self.set_bad_pixels_to_nan(data, self.bad_pixels)
        if self.thres_mips is not None:
            self.exclude_mips_frames(data)
        #offset the data and correct for common mode if necessary
        data = data - self.offset_raw[np.newaxis,:,:,:]
        self.offset_raw = None
        gc.collect()
        if self.comm_mode:
            data = self.get_common_corrected_data(data)
        if self.use_fitted_offset:
            data -= self.offset_fitted[np.newaxis,:,np.newaxis,:]
        avg_over_nreps = self.get_avg_over_nreps(data)
        np.save(os.path.join(self.step_dir, 'rndr_signals.npy'),
                avg_over_nreps)
        #calculate event map and save it
        event_map = self.calc_event_map(avg_over_nreps)
        np.save(os.path.join(self.step_dir, 'event_map.npy'),
                event_map)
        np.save(os.path.join(self.step_dir, 'sum_of_event_signals.npy'),
                self.get_sum_of_event_signals(event_map))
        np.save(os.path.join(self.step_dir, 'sum_of_event_counts.npy'),
                self.get_sum_of_event_counts(event_map))
                
    def calc_event_map(self, avg_over_nreps):
        print('Finding events')
        threshold_map = self.noise_fitted * self.thres_event
        events = avg_over_nreps > threshold_map[np.newaxis,:,:]
        signals = avg_over_nreps[events]
        indices = np.transpose(np.where(events))
        print(f'{signals.shape[0]} events found')
        event_array = np.concatenate(
            (indices, signals[:,np.newaxis]),
              axis = 1
            )
        event_map = np.zeros((64,64), dtype = object)
        event_map.fill([])
        for entry in event_array:
            row = int(entry[1])
            column = int(entry[2])
            signal = entry[3]
            event_map[row][column] = np.append(
                event_map[row][column], signal
                )
        return event_map

    def get_sum_of_event_signals(self, event_map):
        sum_of_events = np.zeros((self.row_size,self.column_size))
        for row in range(self.row_size):
            for column in range(self.column_size):
                sum_of_events[row][column] = sum(event_map[row][column])
        return sum_of_events
    
    def get_sum_of_event_counts(self, event_map):
        sum_of_events = np.zeros((self.row_size,self.column_size))
        for row in range(self.row_size):
            for column in range(self.column_size):
                sum_of_events[row][column] = len(event_map[row][column])
        return sum_of_events