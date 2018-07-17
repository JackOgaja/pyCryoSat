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

    def __init__(self):
        """
        create the class instance
        """

    def read_dbl(file_input, baseline):
        """
        read .dbl file 
        """

        l2ia = [ csa.l2Iarray(file_input, 4, kj) 
             for kj in np.arange(1,131) ]

        ddict = {'Day':l2ia[0]}
        ddict['Sec']                               = l2ia[1]
        ddict['Micsec']                            = l2ia[2]
        ddict['USO_Correction_factor']             = l2ia[3]
        ddict['Mode_id']                           = l2ia[4]
        ddict['Src_Seq_Counter']                   = l2ia[5]
        ddict['Instrument_config']                 = l2ia[6]
        ddict['Rec_Counter']                       = l2ia[7]
        ddict['Lat']                               = l2ia[8]
        ddict['Lon']                               = l2ia[9]
        ddict['Altitude']                          = l2ia[10]
        ddict['Altitude_rate']                     = l2ia[11]
        ddict['Satellite_velocity']                = l2ia[12]
        ddict['Real_beam']                         = l2ia[13]
        ddict['Baseline']                          = l2ia[14]
        ddict['Star_Tracker_id']                   = l2ia[15]
        ddict['Roll_angle']                        = l2ia[16]
        ddict['Pitch_anlge']                       = l2ia[17]
        ddict['Yaw_angle']                         = l2ia[18]
        ddict['Level2_MCD']                        = l2ia[19]
        ddict['Height']                            = l2ia[20]
        ddict['Height_2']                          = l2ia[21]
        ddict['Hieght_3']                          = l2ia[22]
        ddict['Sigma_0']                           = l2ia[23]
        ddict['Sigma_0_2']                         = l2ia[24]
        ddict['Sigma_0_3']                         = l2ia[25]
        ddict['Significant_Wave_Height']           = l2ia[26]
        ddict['Peakiness']                         = l2ia[27]
        ddict['Retracked_range_correction']        = l2ia[28]
        ddict['Retracked_range_correction_2']      = l2ia[29]
        ddict['Retracked_range_correction_3']      = l2ia[30]
        ddict['Retracked_sig0_correction']         = l2ia[31]
        ddict['Retracked_sig0_correction_2']       = l2ia[32]
        ddict['Retracked_sig0_correction_3']       = l2ia[33]
        ddict['Retracked_quality_1']               = l2ia[34]
        ddict['Retracked_quality_2']               = l2ia[35]
        ddict['Retracked_quality_3']               = l2ia[36]
        ddict['Retracked_3']                       = l2ia[37]
        ddict['Retracked_4']                       = l2ia[38]
        ddict['Retracked_5']                       = l2ia[39]
        ddict['Retracked_6']                       = l2ia[40]
        ddict['Retracked_7']                       = l2ia[41]
        ddict['Retracked_8']                       = l2ia[42]
        ddict['Retracked_9']                       = l2ia[43]
        ddict['Retracked_10']                      = l2ia[44]
        ddict['Retracked_11']                      = l2ia[45]
        ddict['Retracked_12']                      = l2ia[46]
        ddict['Retracked_13']                      = l2ia[47]
        ddict['Retracked_14']                      = l2ia[48]
        ddict['Retracked_15']                      = l2ia[49]
        ddict['Retracked_16']                      = l2ia[50]
        ddict['Retracked_17']                      = l2ia[51]
        ddict['Retracked_18']                      = l2ia[52]
        ddict['Retracked_19']                      = l2ia[53]
        ddict['Retracked_20']                      = l2ia[54]
        ddict['Retracked_21']                      = l2ia[55]
        ddict['Retracked_22']                      = l2ia[56]
        ddict['Retracked_23']                      = l2ia[57]
        ddict['Power_echo_shape']                  = l2ia[58]
        ddict['Beam_behaviour_param']              = l2ia[59]
        ddict['Cross_track_angle']                 = l2ia[60]
        ddict['Cross_track_angle_correction']      = l2ia[61]
        ddict['Coherence']                         = l2ia[62]
        ddict['Interpolated_Ocean_Height']         = l2ia[63]
        ddict['Freeboard']                         = l2ia[64]
        ddict['Surface_Height_Anomaly']            = l2ia[65]
        ddict['Interpolated_sea_surface_h_anom']   = l2ia[66]
        ddict['Ocean_height_interpolation_error']  = l2ia[67]
        ddict['Interpolated_forward_records']      = l2ia[68]
        ddict['Interpolated_backward_records']     = l2ia[69]
        ddict['Time_interpolated_fwd_recs']        = l2ia[70]
        ddict['Time_interpolated_bkwd_recs']       = l2ia[71]
        ddict['Interpolation_error_flag']          = l2ia[72]
        ddict['Measurement_mode']                  = l2ia[73]
        ddict['Quality_flags']                     = l2ia[74]
        ddict['Retracker_flags']                   = l2ia[75]
        ddict['Height_status']                     = l2ia[76]
        ddict['Freeboard_status']                  = l2ia[77]
        ddict['Average_echoes']                    = l2ia[78]
        ddict['Wind_speed']                        = l2ia[79]
        ddict['Ice_concentration']                 = l2ia[80]
        ddict['Snow_depth']                        = l2ia[81]
        ddict['Snow_density']                      = l2ia[82]
        ddict['Discriminator']                     = l2ia[83]
        ddict['SARin_discriminator_1']             = l2ia[84]
        ddict['SARin_discriminator_2']             = l2ia[85]
        ddict['SARin_discriminator_3']             = l2ia[86]
        ddict['SARin_discriminator_4']             = l2ia[87]
        ddict['SARin_discriminator_5']             = l2ia[88]
        ddict['SARin_discriminator_6']             = l2ia[89]
        ddict['SARin_discriminator_7']             = l2ia[90]
        ddict['SARin_discriminator_8']             = l2ia[91]
        ddict['SARin_discriminator_9']             = l2ia[92]
        ddict['SARin_discriminator_10']            = l2ia[93]
        ddict['Discriminator_flags']               = l2ia[94]
        ddict['Attitude']                          = l2ia[95]
        ddict['Azimuth']                           = l2ia[96]
        ddict['Slope_doppler_correction']          = l2ia[97]
        ddict['Satellite_latitude']                = l2ia[98]
        ddict['Satellite_longitude']               = l2ia[99]
        ddict['Ambiguity']                         = l2ia[100]
        ddict['MSS_model']                         = l2ia[101]
        ddict['Geoid_model']                       = l2ia[102]
        ddict['ODLE_model']                        = l2ia[103]
        ddict['DEM_elevation']                     = l2ia[104]
        ddict['DEM_id']                            = l2ia[105]
        ddict['Dry_correction']                    = l2ia[106]
        ddict['Wet_correction']                    = l2ia[107]
        ddict['IB_correction']                     = l2ia[108]
        ddict['DAC_correction']                    = l2ia[109]
        ddict['Ionospheric_GIM_correction']        = l2ia[110]
        ddict['Ionospheric_model_correction']      = l2ia[111]
        ddict['Ocean_tide']                        = l2ia[112]
        ddict['LPE_Ocean_tide']                    = l2ia[113]
        ddict['Ocean_loading_tide']                = l2ia[114]
        ddict['Solid_earth_tide']                  = l2ia[115]
        ddict['Geocentric_polar_tide']             = l2ia[116]
        ddict['Surface_type']                      = l2ia[117]
        ddict['Correction_status']                 = l2ia[118]
        ddict['Correction_error']                  = l2ia[119]
        ddict['SS_Bias_correction']                = l2ia[120]
        ddict['Doppler_rc']                        = l2ia[121]
        ddict['TR_instrument_rc']                  = l2ia[122]
        ddict['R_instrument_rc']                   = l2ia[123]
        ddict['TR_instrument_gain_correction']     = l2ia[124]
        ddict['R_instrument_gain_correction']      = l2ia[125]
        ddict['Internal_phase_correction']         = l2ia[126]
        ddict['External_phase_correction']         = l2ia[127]
        ddict['Noise_power_measurement']           = l2ia[128]
        ddict['Phase_slope_correction']            = l2ia[129]

        return ddict 

