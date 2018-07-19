# -*- coding: utf-8 -*-

__all__ = [ 'read_data', 'write_data' ]

__version__ = '0.1.0'
__description__ = 'pycryosat'
__author__ = 'Jack Ogaja  <jack_ogaja@brown.edu> '
__license__ = 'MIT'

#------------------------#

import numpy as np
import lib.csarray as csa 

#------------------------#

class read_data(object):
    """
    Attributes.
    : read_file
    Reads .dbl/.hdr input files
    """

    __FieldsStrings = [
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

    def __init__(self):
        """
        create the class instance
        """
        pass

    @property
    def fields(self):
        return read_data.__FieldsStrings
#        return "[" + ", ".join( str(x) for x in __FieldsStrings) + "]"
#         for x in __FieldsStrings:
#             print(x)
        
#+    @fields.setter
#+    def fields(self, fields):
#+        pass 

#    def __getattribute__(self, attr):
#        if attr == 'fields':
#           if len(__FieldsStrings) == 130:
#              return read_data.__FieldsStrings
#           else:
#              raise AttributeError( "Attribute fields is not properly defined" )
#        else:
#           raise AttributeError

    def read_dbl(file_input, baseline):
        """
        read .dbl file 
        """

        l2ia = [ csa.l2Iarray(file_input, 4, kj) 
             for kj in np.arange(1,131) ]

        ddict = {}

        for n, dstr in enumerate(read_data.__FieldsStrings):
            if dstr == 'Freeboard':
               ddict[dstr] = l2ia[n]*0.01
            else:
               ddict[dstr] = l2ia[n]

        return ddict 

