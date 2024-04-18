import gc
import os
from datetime import datetime

import numpy as np
from scipy.optimize import curve_fit

from nproan.commonclass import Common

class Filter(Common):
    def __init__(self):
        super().__init__()
        print('Filter object created\nRun load_params()\n~~~~~')

        self.offnoi_dir = None
        self.data = None
        self.offset_raw = None
        self.offset_fitted = None
        self.noise_fitted = None
        self.avg_over_nreps = None
        self.event_map = None

    def load(self, parameters, offnoi_dir):
        self.bin_file = parameters['signal_bin_file']
        self.column_size = parameters['common_column_size']
        self.row_size = parameters['common_row_size']
        self.nreps = parameters['signal_nreps']
        self.key_ints = parameters['common_key_ints']
        self.nframes = parameters['signal_nframes']
        self.bad_pixels = parameters['common_bad_pixels']
        self.comm_mode = parameters['signal_comm_mode']
        self.thres_mips = parameters['signal_thres_mips']
        self.thres_event = parameters['signal_thres_event']
        self.fitted_offset = parameters['signal_fitted_offset']

        print(f'Parameters loaded:\n\
              file: {self.bin_file}\n\
              column_size: {self.column_size}\n\
              row_size: {self.row_size}\n\
              nreps: {self.nreps}\n\
              key_ints: {self.key_ints}\n\
              max_frames: {self.nframes}\n\
              bad_pixels: {self.bad_pixels}\n\
              comm_mode: {self.comm_mode}\n\
              thres_mips: {self.thres_mips}\n\
              thres_event: {self.thres_event}\n\
              fitted_offset: {self.fitted_offset}')
        
        print('Loading offnoi data\n')
        try:
            #offset_raw is quite big. deleted after use
            self.offset_raw = np.load(
                os.path.join(offnoi_dir, 'offset_raw.npy')
            )
            self.offset_fitted = np.load(
                os.path.join(offnoi_dir, 'fitted_offset.npy')
            )
            self.noise_fitted = np.load(
                os.path.join(offnoi_dir, 'fitted_noise.npy')
            )
            self.offnoi_dir = offnoi_dir
            self.common_dir = os.path.dirname(offnoi_dir)
        except:
            print('Error loading offnoi data\n')
            return
        print('Offnoi data loaded\nRun readData()\n~~~~~')

    def calculate(self):
        #create the working directory for the filter step
        self.step_dir = os.path.join(
            self.common_dir, f'filter_{self.thres_event}_threshold'
        )
        os.makedirs(self.step_dir, exist_ok=True)
        print(f'Created directory for filter step: {self.step_dir}')

        if self.data is None:
            self.read_data()
        #omit bad pixels and mips frames
        if self.bad_pixels is not None:
            self._set_bad_pixels(self.bad_pixels)
        if self.thres_mips is not None:
            self._exclude_mips_frames(self.thres_mips)
        #offset the data and correct for common mode if necessary
        self.data -= self.offset_raw[np.newaxis,:,:,:]
        self.offset_raw = None
        gc.collect()
        if self.comm_mode:
            self._correct_common_mode()
        if self.fitted_offset:
            self.data -= self.offset_fitted[np.newaxis,:,np.newaxis,:]
        self._calc_avg_over_nreps()
        np.save(os.path.join(self.step_dir, 'rndr_signals.npy'),
                self.avg_over_nreps)
        #calculate event map and save it
        self._calc_event_map()
        np.save(os.path.join(self.step_dir, 'event_map.npy'),
                self.event_map)
        np.save(os.path.join(self.step_dir, 'sum_of_event_signals.npy'),
                self._get_sum_of_event_signals())
        np.save(os.path.join(self.step_dir, 'sum_of_event_counts.npy'),
                self._get_sum_of_event_counts())
                
    def _calc_event_map(self):
        print('Finding events')
        threshold_map = self.noise_fitted * self.thres_event
        events = self.avg_over_nreps > threshold_map[np.newaxis,:,:]
        signals = self.avg_over_nreps[events]
        indices = np.transpose(np.where(events))
        print(f'{signals.shape[0]} events found')
        event_array = np.concatenate(
            (indices, signals[:,np.newaxis]),
              axis = 1
            )
        self.event_map = np.zeros((64,64), dtype = object)
        self.event_map.fill([])
        for entry in event_array:
            row = int(entry[1])
            column = int(entry[2])
            signal = entry[3]
            self.event_map[row][column] = np.append(
                self.event_map[row][column], signal
                )

    def _get_sum_of_event_signals(self):
        sum_of_events = np.zeros((self.row_size,self.column_size))
        for row in range(self.row_size):
            for column in range(self.column_size):
                sum_of_events[row][column] = sum(self.event_map[row][column])
        return sum_of_events
    
    def _get_sum_of_event_counts(self):
        sum_of_events = np.zeros((self.row_size,self.column_size))
        for row in range(self.row_size):
            for column in range(self.column_size):
                sum_of_events[row][column] = len(self.event_map[row][column])
        return sum_of_events