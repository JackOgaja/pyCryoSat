# -*- coding: utf-8 -*-

__all__ = [ 'fields', 'read_rdbl' ]

__version__ = '0.1.0'
__description__ = 'pycryosat'
__author__ = 'Jack Ogaja  <jack_ogaja@brown.edu> '
__license__ = 'MIT'

#------------------------#

import os, sys, inspect
import numpy as np
import lib.csarray as csa 
import itertools

#------------------------#

class pycryosat(object):
    """
    Attributes.
    : read_file
    Reads .dbl/.hdr input files
    """

    __FieldsStrings_l2i = [
    'Day', 'Sec', 'Micsec', 'USO_Correction_factor', 'Mode_id', 
    'Src_Seq_Counter', 'Instrument_config', 'Rec_Counter', 
    'Lat', 'Lon', 'Altitude', 'Altitude_rate', 'Satellite_velocity', 
    'Real_beam', 'Baseline', 'Star_Tracker_id', 'Roll_angle', 
    'Pitch_anlge', 'Yaw_angle', 'Level2_MCD', 'Height', 'Height_2', 
    'Hieght_3', 'Sigma_0', 'Sigma_0_2', 'Sigma_0_3', 'Significant_Wave_Height', 
    'Peakiness', 'Retracked_range_correction', 'Retracked_range_correction_2', 
    'Retracked_range_correction_3', 'Retracked_sig0_correction', 
    'Retracked_sig0_correction_2', 'Retracked_sig0_correction_3', 
    'Retracked_quality_1', 'Retracked_quality_2', 'Retracked_quality_3', 
    'Retracked_3', 'Retracked_4', 'Retracked_5', 'Retracked_6', 
    'Retracked_7', 'Retracked_8', 'Retracked_9', 'Retracked_10', 
    'Retracked_11', 'Retracked_12', 'Retracked_13', 'Retracked_14', 
    'Retracked_15', 'Retracked_16', 'Retracked_17', 'Retracked_18', 
    'Retracked_19', 'Retracked_20', 'Retracked_21', 'Retracked_22', 
    'Retracked_23', 'Power_echo_shape', 'Beam_behaviour_param', 
    'Cross_track_angle', 'Cross_track_angle_correction', 'Coherence', 
    'Interpolated_Ocean_Height', 'Freeboard', 'Surface_Height_Anomaly', 
    'Interpolated_sea_surface_h_anom', 'Ocean_height_interpolation_error', 
    'Interpolated_forward_records', 'Interpolated_backward_records', 
    'Time_interpolated_fwd_recs', 'Time_interpolated_bkwd_recs', 
    'Interpolation_error_flag', 'Measurement_mode', 'Quality_flags', 
    'Retracker_flags', 'Height_status', 'Freeboard_status', 
    'Average_echoes', 'Wind_speed', 'Ice_concentration', 'Snow_depth', 
    'Snow_density', 'Discriminator', 'SARin_discriminator_1', 
    'SARin_discriminator_2', 'SARin_discriminator_3', 'SARin_discriminator_4', 
    'SARin_discriminator_5', 'SARin_discriminator_6', 'SARin_discriminator_7', 
    'SARin_discriminator_8', 'SARin_discriminator_9', 'SARin_discriminator_10', 
    'Discriminator_flags', 'Attitude', 'Azimuth', 'Slope_doppler_correction', 
    'Satellite_latitude', 'Satellite_longitude', 'Ambiguity', 'MSS_model', 
    'Geoid_model', 'ODLE_model', 'DEM_elevation', 'DEM_id', 'Dry_correction', 
    'Wet_correction', 'IB_correction', 'DAC_correction', 'Ionospheric_GIM_correction', 
    'Ionospheric_model_correction', 'Ocean_tide', 'LPE_Ocean_tide', 
    'Ocean_loading_tide', 'Solid_earth_tide', 'Geocentric_polar_tide', 
    'Surface_type', 'Correction_status', 'Correction_error', 
    'SS_Bias_correction', 'Doppler_rc', 'TR_instrument_rc', 'R_instrument_rc', 
    'TR_instrument_gain_correction', 'R_instrument_gain_correction', 
    'Internal_phase_correction', 'External_phase_correction', 
    'Noise_power_measurement', 'Phase_slope_correction' ] 

    __FieldsFactors_l2i = [
    1e-7, 1e-7, 1e-3, 1e-3, 1/1e3, 1/1e6, 1/1e6, 1e-7, 1e-7, 1e-7,
    1e-3, 1e-3, 1e-3, 1/100, 1/100, 1/100, 1e-3, 1e-2, 1e-3, 1e-3,
    1e-3, 1e-2, 1e-2, 1e-2, 1e-2,   1,     1,    1,    1,    1,  
    1,    1,    1,     1,   1,      1,     1,    1,    1,    1, 
    1,    1,    1,     1,   1,      1,     1,    1,    1,    1, 
    1,    1,    1,     1,   1,      1,     1,    1,    1,    1, 
    1,    1,    1,     1,   1,      1,     1,    1,    1,    1, 
    1,    1,    1,     1,   1,      1,     1,    1,    1,    1, 
    1,    1,    1,     1,   1,      1,     1,    1,    1,    1, 
    1,    1,    1,     1,   1,      1,     1,    1,    1,    1, 
    1,    1,    1,     1,   1,      1,     1,    1,    1,    1, 
    1,    1,    1,     1,   1,      1,     1,    1,    1,    1, 
    1,    1,    1,     1,   1,      1,     1,    1,    1,    1   ]  

    def __init__(self, file_input, baseline):
        """
        create the class instance
        """
        self.input = file_input
        self.base = baseline

    @property
    def fields(self):
        return self.__FieldsStrings_l2i
        
    def __readFile(self):
        """
        read .dbl file 
        """
        l2ia = [ csa.l2Iarray(self.input, self.base, kj)*kf 
             for kj,kf in np.nditer([np.arange(1,131), self.__FieldsFactors_l2i]) ]

        return l2ia

    @property
    def readToDict(self):
        """
        read to dictionary 
        """
        
        rn = self.__readFile()  
        ddict = {}

        for n, dstr in enumerate(self.__FieldsStrings_l2i):

            ddict[dstr] = rn[n]

        return ddict 

    @property
    def readToCsvFile(self):
        """
        Write out data to a file
        """
        fname = 'pyOut.csv'
        rd = self.__readFile()[0:11]

        #- python 3X.
        data = list(map(list, zip(*rd))) 

        try:
              import csv
              try:
                  with open(fname, 'w') as f:
                        fn = self.__FieldsStrings_l2i[0:11]
                        out_file = csv.DictWriter(f, fieldnames=fn, delimiter=';')
                        out_file.writeheader()
                        for rw in data:
                             out_file.writerow({fn[0]: rw[0], fn[1]: rw[1], fn[2]: rw[2],
                                                fn[3]: rw[3], fn[4]: rw[4], fn[5]: rw[5],
                                                fn[6]: rw[6], fn[7]: rw[7], fn[8]: rw[8],
                                                fn[9]: rw[9], fn[10]: rw[10]}) #, fn[11]: rw[11],
#                                                fn[12]: rw[12], fn[13]: rw[13], fn[14]: rw[14]})


              except csv.Error as e:
                   sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))

        except ImportError:
               raise ImportError('cannot import csv module')


