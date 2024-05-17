import gc
import os
from datetime import datetime

import numpy as np

import logger
import analysis_funcs as af
import analysis as an
import params as pm
class OffNoi():

    _logger = logger.Logger('nproan-offnoi', 'debug').get_logger()

    def __init__(self, prm_file):
        self.load(prm_file)
        self._logger.info('OffNoi object created')

    def load(self, prm_file):
        self.prm = pm.Params(prm_file)
        parameters = self.prm.get_dict()
        #common parameters
        self.results_dir = parameters['common_results_dir']
        self.column_size = parameters['common_column_size']
        self.row_size = parameters['common_row_size']
        self.key_ints = parameters['common_key_ints']
        self.bad_pixels = parameters['common_bad_pixels']
        #offnoi parameters
        self.bin_file = parameters['offnoi_bin_file']
        self.nreps = parameters['offnoi_nreps']
        self.nframes = parameters['offnoi_nframes']
        self.nreps_eval = parameters['offnoi_nreps_eval']
        self.comm_mode = parameters['offnoi_comm_mode']
        self.thres_mips = parameters['offnoi_thres_mips']
        self.thres_bad_frames = parameters['offnoi_thres_bad_frames']
        self.thres_bad_slopes = parameters['offnoi_thres_bad_slopes']

        #directories, they will be created in calculate()
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        filename = self.get_bin_file_name()[:-4]
        self.common_dir = os.path.join(
            self.results_dir, timestamp + '_' + filename)
        self.step_dir = os.path.join(self.common_dir, 
            f'offnoi_{self.nreps}reps_{self.nframes}frames')

        self._logger.info(f'Parameters loaded:')
        self.prm.print_contents()

    def calculate(self):   
        #reate the directory for the data
        os.makedirs(self.common_dir, exist_ok=True)
        self._logger.info(f'Created common directory for data: {self.common_dir}')
        #now, create the working directory for the offnoi step
        os.makedirs(self.step_dir, exist_ok=True)
        self._logger.info(f'Created working directory for offnoi step: {self.step_dir}')
        # and save the parameter file there
        self.prm.save(os.path.join(self.step_dir, 'parameters.json'))

        data = self.get_data()
        gc.collect()
        #delete nreps_eval from data
        if self.nreps_eval:
            data = self.exclude_nreps_eval(data)
            self._logger.debug(f'Shape of data: {data.shape}')
        #set values of all frames and nreps of bad pixels to nan
        if self.bad_pixels:
            data = self.set_bad_pixellist_to_nan(data)
        #deletes bad frames from data
        if self.thres_bad_frames != 0:
            data = self.exclude_bad_frames(data)
            self._logger.debug(f'Shape of data: {data.shape}')
        #deletes frames with mips above threshold from data
        if self.thres_mips != 0:
            data = self.exclude_mips_frames(data)
            self._logger.debug(f'Shape of data: {data.shape}')
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
            self.correct_common_mode(data)
        #calculate rndr signals and save it
        avg_over_nreps = self.get_avg_over_nreps(data)
        np.save(os.path.join(self.step_dir, 'rndr_signals.npy'),
                avg_over_nreps)
        #calculate fitted offset and noise and save it (including fit errors)
        fit_unbinned = cm.get_unbinned_fit_gauss(avg_over_nreps)
        fit_curve_fit = cm.get_fit_gauss(avg_over_nreps)
        np.save(os.path.join(self.step_dir, 'offnoi_fit_unbinned.npy'), fit_unbinned)
        np.save(os.path.join(self.step_dir, 'offnoi_fit.npy'), fit_curve_fit)
        if self.thres_bad_slopes != 0:
            bad_slopes_pos, bad_slopes_data, bad_slopes_value = self.get_bad_slopes(data)
            np.save(os.path.join(self.step_dir, 'bad_slopes_pos.npy'), bad_slopes_pos)
            np.save(os.path.join(self.step_dir, 'bad_slopes_data.npy'), bad_slopes_data)
            np.save(os.path.join(self.step_dir, 'bad_slopes_value.npy'), bad_slopes_value)

class Filter():

    _logger = logger.Logger('nproan-filter', 'debug').get_logger()

    def __init__(self, prm_file, offnoi_dir):
        self.load(prm_file, offnoi_dir)
        self._logger.info('Filter object created')

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
        self.nreps_eval = parameters['filter_nreps_eval']
        self.comm_mode = parameters['filter_comm_mode']
        self.thres_mips = parameters['filter_thres_mips']
        self.thres_event = parameters['filter_thres_event']
        self.use_fitted_offset = parameters['filter_use_fitted_offset']
        self.thres_bad_frames = parameters['filter_thres_bad_frames']
        self.thres_bad_slopes = parameters['filter_thres_bad_slopes']

        #directories
        #set self.common_dir to the parent directory of offnoi_dir
        self.common_dir = os.path.dirname(offnoi_dir)
        self.step_dir = None
        
        self._logger.info(f'Parameters loaded:')
        self.prm.print_contents()
        
        self._logger.info('Checking parameters in offnoi directory')
        #look for a json file in the offnoi directory 
        if (not self.prm.same_common_params(offnoi_dir)) \
            or (not self.prm.same_offnoi_params(offnoi_dir)):
            self._logger.error('Parameters in offnoi directory do not match')
            return
        try:
            #offset_raw is quite big. deleted after use
            self.offset_raw = cm.get_array_from_file(
                offnoi_dir, 'offset_raw.npy')
            self._logger.debug(self.offset_raw.shape)
            if self.offset_raw is None:
                self._logger.error('Error loading offset_raw data\n')
                return
            self.offset_fitted = cm.get_array_from_file(
                offnoi_dir, 'offnoi_fit.npy'
            )[1]
            if self.offset_fitted is None:
                self._logger.error('Error loading fitted_offset data\n')
                return
            self.noise_fitted = cm.get_array_from_file(
                offnoi_dir, 'offnoi_fit.npy'
            )[2]
            if self.noise_fitted is None:
                self._logger.error('Error loading fitted_noise data\n')
                return
            self.offnoi_dir = offnoi_dir
            self.common_dir = os.path.dirname(offnoi_dir)
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            self.step_dir = os.path.join(
                self.common_dir,f'filter_{self.thres_event}_threshold'
            )
            self._logger.debug(self.step_dir)
        except:
            raise ValueError('Error loading offnoi data\n')

    def calculate(self):
        #create the working directory for the filter step
        os.makedirs(self.step_dir, exist_ok=True)
        self._logger.info(f'Created directory for filter step: {self.step_dir}')
        # and save the parameter file there
        self.prm.save(os.path.join(self.step_dir, 'parameters.json'))

        data = self.get_data()
        gc.collect()
        if self.nreps_eval:
            data = self.exclude_nreps_eval(data)
            self._logger.debug(f'Shape of data: {data.shape}')
        #omit bad pixels and mips frames
        if self.bad_pixels:
            data = self.set_bad_pixellist_to_nan(data)
        if self.thres_bad_frames != 0:
            data = self.exclude_bad_frames(data)
            self._logger.debug(f'Shape of data: {data.shape}')
        if self.thres_mips != 0:
            data = self.exclude_mips_frames(data)
            self._logger.debug(f'Shape of data: {data.shape}')
        #offset the data and correct for common mode if necessary
        data = data - self.offset_raw[np.newaxis,:,:,:]
        self.offset_raw = None
        gc.collect()
        if self.comm_mode:
            self.correct_common_mode(data)
        if self.use_fitted_offset:
            #take care here, offset fitted can contain np.nan
            data -= np.nan_to_num(self.offset_fitted[np.newaxis,:,np.newaxis,:])
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
        if self.thres_bad_slopes != 0:
            bad_slopes_pos, bad_slopes_data, bad_slopes_value = self.get_bad_slopes(data)
            np.save(os.path.join(self.step_dir, 'bad_slopes_pos.npy'), bad_slopes_pos)
            np.save(os.path.join(self.step_dir, 'bad_slopes_data.npy'), bad_slopes_data)
            np.save(os.path.join(self.step_dir, 'bad_slopes_value.npy'), bad_slopes_value)

class Gain():

    _logger = logger.Logger('nproan-gain', 'debug').get_logger()

    def __init__(self, prm_file, filter_dir):
        self.load(prm_file, filter_dir)
        self._logger.info('Gain object created')

    def load(self, prm_file, filter_dir):
        self.prm = pm.Params(prm_file)
        parameters = self.prm.get_dict()
        #common parameters
        self.results_dir = parameters['common_results_dir']
        self.column_size = parameters['common_column_size']
        self.row_size = parameters['common_row_size']
        self.key_ints = parameters['common_key_ints']
        self.bad_pixels = parameters['common_bad_pixels']

        #gain parameters
        self.nreps = parameters['filter_nreps']
        self.nframes = parameters['filter_nframes']
        self.min_signals = parameters['gain_min_signals']

        self._logger.info(f'Parameters loaded:')
        self.prm.print_contents()
        
        self._logger.info('Checking parameters in filter directory')
        #look for a json file in the filter directory
        if (not self.prm.same_common_params(filter_dir)) \
            or (not self.prm.same_offnoi_params(filter_dir) \
            or (not self.prm.same_filter_params(filter_dir))):
            self._logger.error('Parameters in filter directory do not match')
            return
        try:
            self.event_map = cm.get_array_from_file(
                filter_dir, 'event_map.npy')
            #set the directory where the filter data is stored
            self.filter_dir = filter_dir
            #this is the parent directory. data from this step is stored there
            self.common_dir = os.path.dirname(filter_dir)
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            self.step_dir = os.path.join(
                self.common_dir, timestamp + f'_gain_{self.min_signals}_minsignals'
            )
        except:
            self._logger.error('Error loading filter data\n')
            return
        self._logger.info('Filter data loaded\n')

    def calculate(self):
        #create the working directory for the gain step
        self.step_dir = os.path.join(self.common_dir, 
                                     f'gain_{self.min_signals}_min_signals')
        os.makedirs(self.step_dir, exist_ok=True)
        self._logger.info(f'Created directory for gain step: {self.step_dir}')
        # and save the parameter file there
        self.prm.save(os.path.join(self.step_dir, 'parameters.json'))
        
        fits = self.get_gain_fit(self.event_map)
        np.save(os.path.join(self.step_dir, 'fit_mean.npy'), fits[0])
        np.save(os.path.join(self.step_dir, 'fit_sigma.npy'), fits[1])
        np.save(os.path.join(self.step_dir, 'fit_mean_error.npy'), fits[2])
        np.save(os.path.join(self.step_dir, 'fit_sigma_error.npy'), fits[3])