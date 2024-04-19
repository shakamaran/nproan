import gc
import os
from datetime import datetime

import numpy as np
from scipy.optimize import curve_fit

import common as cm
import params as pm

class OffNoi(cm.Common):
    def __init__(self, prm_file):
        self.load(prm_file)
        print('OffNoi object created')

    def load(self, prm_file):
        prm = pm.Params(prm_file)
        parameters = prm.get_dict()
        #common parameters
        self.results_dir = parameters['common_results_dir']
        self.column_size = parameters['common_column_size']
        self.row_size = parameters['common_row_size']
        self.key_ints = parameters['common_key_ints']
        self.bad_pixels = parameters['common_bad_pixels']
        #offnoi parameters
        self.bin_file = parameters['dark_bin_file']
        self.nreps = parameters['dark_nreps']
        self.nframes = parameters['dark_nframes']
        self.comm_mode = parameters['dark_comm_mode']
        self.thres_mips = parameters['dark_thres_mips']

        #directories, they will be created in calculate()
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        filename = self.get_bin_file_name()[:-4]
        self.common_dir = os.path.join(
            self.results_dir, timestamp + '_' + filename)
        self.step_dir = os.path.join(self.common_dir, 
            f'offnoi_{self.nreps}reps_{self.nframes}frames')

        print(f'Parameters loaded:)')
        prm.print_contents()

    def calculate(self):   
        #reate the directory for the data
        os.makedirs(self.common_dir, exist_ok=True)
        print(f'Created common directory for data: {self.common_dir}')
        #now, create the working directory for the offnoi step
        os.makedirs(self.step_dir, exist_ok=True)
        print(f'Created working directory for offnoi step: {self.step_dir}')

        data = self.get_data()
        #omit bad pixels and mips frames
        if self.bad_pixels is not None:
            self.set_bad_pixels_to_nan(data, self.bad_pixels)
        if self.thres_mips is not None:
            self.exclude_mips_frames(data)
        #calculate offset_raw on the raw data and save it
        avg_over_frames = self.get_avg_over_frames(data)
        np.save(os.path.join(self.step_dir, 'offset_raw.npy'),
                avg_over_frames)
        #calculate offset and save it
        avg_over_frames_and_nreps = self.get_avg_over_frames_and_nreps(data)
        np.save(os.path.join(self.step_dir, 'offset.npy'),
                avg_over_frames_and_nreps)
        #offset the data and correct for common mode if necessary
        data -= avg_over_frames[np.newaxis,:,:,:]
        del avg_over_frames
        gc.collect()
        if self.comm_mode is True:
            data = self.get_common_corrected_data(data)
        #calculate rndr signals and save it
        avg_over_nreps = self.get_avg_over_nreps(data)
        np.save(os.path.join(self.step_dir, 'rndr_signals.npy'),
                avg_over_nreps)
        #calculate fitted offset and noise and save it (including fit errors)
        offnoi = self.get_fitted_offnoi(avg_over_nreps)
        np.save(os.path.join(self.step_dir, 'fitted_offset.npy'), offnoi[0])
        np.save(os.path.join(self.step_dir, 'fitted_noise.npy'), offnoi[1])
        np.save(
            os.path.join(self.step_dir, 'fitted_offset_error.npy'), offnoi[2]
        )
        np.save(
            os.path.join(self.step_dir, 'fitted_noise_error.npy'), offnoi[3]
        )