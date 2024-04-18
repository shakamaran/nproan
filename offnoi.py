import gc
import os
from datetime import datetime

import numpy as np
from scipy.optimize import curve_fit

from nproan.commonclass import Common

class OffNoi(Common):
    def __init__(self):
        super().__init__()
        print('OffNoi object created\nRun load_params()\n~~~~~')
        
        self.data = None
        self.common_stats = None
        self.avg_over_frames = None
        self.avg_over_nreps = None
        self.avg_over_frames_and_nreps = None

    def load(self, parameters, results_dir):
        self.bin_file = parameters['dark_bin_file']
        self.column_size = parameters['common_column_size']
        self.row_size = parameters['common_row_size']
        self.nreps = parameters['dark_nreps']
        self.key_ints = parameters['common_key_ints']
        self.nframes = parameters['dark_nframes']
        self.bad_pixels = parameters['common_bad_pixels']
        self.comm_mode = parameters['dark_comm_mode']
        self.thres_mips = parameters['dark_thres_mips']

        self.results_dir = results_dir

        print(f'Parameters loaded:\n\
              file: {self.bin_file}\n\
              column_size: {self.column_size}\n\
              row_size: {self.row_size}\n\
              nreps: {self.nreps}\n\
              key_ints: {self.key_ints}\n\
              max_frames: {self.nframes}\n\
              bad_pixels: {self.bad_pixels}\n\
              comm_mode: {self.comm_mode}\n\
              thres_mips: {self.thres_mips}')
        
        print('Run read_data()\n~~~~~')

    def calculate(self):
        '''
        first, create the directory for the data
        '''
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        filename = self._get_bin_file_name()
        print(f'filename: {filename}')
        self.common_dir = os.path.join(
            self.results_dir, timestamp + '_' + filename + '_data'
        )
        os.makedirs(self.common_dir, exist_ok=True)
        print(f'Created common directory for data: {self.common_dir}')

        #now, create the working directory for the offnoi step
        self.step_dir = os.path.join(
            self.common_dir, 
            f'offnoi_{self.nreps}reps_{self.nframes}frames'
        )
        os.makedirs(self.step_dir, exist_ok=True)
        print(f'Created working directory for offnoi step: {self.step_dir}')

        if self.data is None:
            self.read_data()
        
        '''
        TODO:
        - slopes
        '''
        #omit bad pixels and mips frames
        if self.bad_pixels is not None:
            self._set_bad_pixels(self.bad_pixels)
        if self.thres_mips is not None:
            self._exclude_mips_frames(self.thres_mips)
        #calculate offset_raw on the raw data and save it
        self._calc_avg_over_frames()
        np.save(os.path.join(self.step_dir, 'offset_raw.npy'),
                self.avg_over_frames)
        #calculate offset and save it
        self._calc_avg_over_frames_and_nreps()
        np.save(os.path.join(self.step_dir, 'offset.npy'),
                self.avg_over_frames_and_nreps)
        #offset the data and correct for common mode if necessary
        self.data -= self.avg_over_frames[np.newaxis,:,:,:]
        self.avg_over_frames = None
        gc.collect()
        if self.comm_mode is True:
            self._correct_common_mode()
        #calculate rndr signals and save it
        self._calc_avg_over_nreps()
        np.save(os.path.join(self.step_dir, 'rndr_signals.npy'),
                self.avg_over_nreps)
        self._calc_avg_over_frames_and_nreps()
        #calculate fitted offset and noise and save it (including fit errors)
        offnoi = self._get_fitted_offnoi()
        np.save(os.path.join(self.step_dir, 'fitted_offset.npy'), offnoi[0])
        np.save(os.path.join(self.step_dir, 'fitted_noise.npy'), offnoi[1])
        np.save(
            os.path.join(self.step_dir, 'fitted_offset_error.npy'), offnoi[2]
        )
        np.save(
            os.path.join(self.step_dir, 'fitted_noise_error.npy'), offnoi[3]
        )