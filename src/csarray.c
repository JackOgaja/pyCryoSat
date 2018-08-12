/*--functions for reading ESA's CryoSat2 data sets--*/

/*==============
These functions use I/O libraries developed by UCL/MSSL, London
I/O library author: (C) CryoSat2 Software Team, 
                   Mullard Space Science Lab, UCL, London
I/O library version: 2.3
================*/

/*
Jack Ogaja, 
Brown University,
20180620
See LICENSE.md for copyright notice
*/

/*-- 
  Make NUMPY API calls available in this module 
--*/
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pycryo_ARRAY_API

#include "csarray.h"

#define FP field_properties

/*-- determine the sizes of different fields --*/
uint8_t
fieldSize( FIELDS field )
{
 switch(field)
    {
      case Field_Unknown:             return 0; 
      case Day:                       return DAY_SIZE; 
      case Sec:                       return SEC_SIZE; 
      case Micsec:                    return MICSEC_SIZE; 
      case USO_Correction_factor:     return USO_CORR_SIZE;
      case Mode_id:                   return MODE_ID_SIZE;
      case Src_Seq_Counter:           return SSC_SIZE;
      case Instrument_config:         return INST_CONFIG_SIZE;
      case Rec_Counter:               return REC_COUNT_SIZE;
      case Lat:                       return LAT_SIZE;
      case Lon:                       return LON_SIZE;
      case Altitude:                  return ALT_SIZE;
      case Altitude_rate:             return ALT_RATE_SIZE;
      case Satellite_velocity:        return SAT_VELOCITY_SIZE; 
      case Real_beam:                 return REAL_BEAM_SIZE;
      case Baseline:                  return BASELINE_SIZE;
      case Star_Tracker_id:           return ST_ID_SIZE;
      case Roll_angle:                return ROLL_ANGLE_SIZE;
      case Pitch_angle:               return PITCH_ANGLE_SIZE;
      case Yaw_angle:                 return YAW_ANGLE_SIZE;
      case Level2_MCD:                return L2_MCD_SIZE;
      case Height:                    return HEIGHT_SIZE;
      case Height_2:                  return HEIGHT_2_SIZE;
      case Height_3:                  return HEIGHT_3_SIZE;
      case Sigma_0:                   return SIG0_SIZE;
      case Sigma_0_2:                 return SIG0_2_SIZE;
      case Sigma_0_3:                 return SIG0_3_SIZE;
      case Significant_Wave_Height:   return SWH_SIZE;
      case Peakiness:                 return PEAKINESS_SIZE;
      case Retracked_range_correction:   return RETRK_RANGE_C_SIZE;
      case Retracked_range_correction_2: return RETRK_RANGE_C_2_SIZE;
      case Retracked_range_correction_3: return RETRK_RANGE_C_3_SIZE;
      case Retracked_sig0_correction:    return RETRK_SIG0_C_SIZE;
      case Retracked_sig0_correction_2:  return RETRK_SIG0_C_2_SIZE;
      case Retracked_sig0_correction_3:  return RETRK_SIG0_C_3_SIZE;
      case Retracked_quality_1:        return RETRK_QUALITY_1_SIZE;
      case Retracked_quality_2:        return RETRK_QUALITY_2_SIZE;
      case Retracked_quality_3:        return RETRK_QUALITY_3_SIZE;
      case Retracked_3:                return RETRK_3_SIZE;
      case Retracked_4:                return RETRK_4_SIZE;
      case Retracked_5:                return RETRK_5_SIZE;
      case Retracked_6:                return RETRK_6_SIZE;
      case Retracked_7:                return RETRK_7_SIZE;
      case Retracked_8:                return RETRK_8_SIZE;
      case Retracked_9:                return RETRK_9_SIZE;
      case Retracked_10:               return RETRK_10_SIZE;
      case Retracked_11:               return RETRK_11_SIZE;
      case Retracked_12:               return RETRK_12_SIZE;
      case Retracked_13:               return RETRK_13_SIZE;
      case Retracked_14:               return RETRK_14_SIZE;
      case Retracked_15:               return RETRK_15_SIZE;
      case Retracked_16:               return RETRK_16_SIZE;
      case Retracked_17:               return RETRK_17_SIZE;
      case Retracked_18:               return RETRK_18_SIZE;
      case Retracked_19:               return RETRK_19_SIZE;
      case Retracked_20:               return RETRK_20_SIZE;
      case Retracked_21:               return RETRK_21_SIZE;
      case Retracked_22:               return RETRK_22_SIZE;
      case Retracked_23:               return RETRK_23_SIZE;
      case Power_echo_shape:           return ECHO_SHAPE_SIZE;
      case Beam_behaviour_param:       return BB_PARAM_SIZE;
      case Cross_track_angle:          return XTRACK_ANGLE_SIZE;
      case Cross_track_angle_correction: return XTRACK_ANGLE_C_SIZE;
      case Coherence:                  return COHERENCE_SIZE;
      case Interpolated_Ocean_Height:  return OCEAN_HT_SIZE; 
      case Freeboard:                  return FREEBOARD_SIZE; 
      case Surface_Height_Anomaly:     return SHA_SIZE; 
      //
      case Interpolated_sea_surface_h_anom:  return SSHA_INTERP_SIZE;
      case Ocean_height_interpolation_error: return INTERP_ERR_SIZE;
      case Interpolated_forward_records:     return INTERP_CNT_FWD_SIZE;
      case Interpolated_backward_records:    return INTERP_CNT_BKWD_SIZE;
      case Time_interpolated_fwd_recs:       return INTERP_TIME_FWD_SIZE;
      case Time_interpolated_bkwd_recs:      return INTERP_TIME_BKWD_SIZE;
      case Interpolation_error_flag:         return INTERP_ERROR_F_SIZE;
      case Measurement_mode:                 return MEAS_MODE_SIZE;
      case Quality_flags:                    return QUALITY_F_SIZE;
      case Retracker_flags:                  return RETRACKER_F_SIZE;
      case Height_status:                    return HT_STATUS_SIZE;
      case Freeboard_status:                 return FREEB_STATUS_SIZE;
      case Average_echoes:                   return N_AVG_SIZE;
      case Wind_speed:                       return WIND_SPEED_SIZE;
      //
      case Ice_concentration:          return ICE_CONC_SIZE;
      case Snow_depth:                 return SNOW_DEPTH_SIZE;
      case Snow_density:               return SNOW_DENSITY_SIZE;
      case Discriminator:              return DISCRIMINATOR_SIZE;
      case SARin_discriminator_1:      return SARIN_DISC_1_SIZE;
      case SARin_discriminator_2:      return SARIN_DISC_2_SIZE;
      case SARin_discriminator_3:      return SARIN_DISC_3_SIZE;
      case SARin_discriminator_4:      return SARIN_DISC_4_SIZE;
      case SARin_discriminator_5:      return SARIN_DISC_5_SIZE;
      case SARin_discriminator_6:      return SARIN_DISC_6_SIZE;
      case SARin_discriminator_7:      return SARIN_DISC_7_SIZE;
      case SARin_discriminator_8:      return SARIN_DISC_8_SIZE;
      case SARin_discriminator_9:      return SARIN_DISC_9_SIZE;
      case SARin_discriminator_10:     return SARIN_DISC_10_SIZE;
      case Discriminator_flags:        return DISCRIM_F_SIZE;
      case Attitude:                   return ATTITUDE_SIZE;
      case Azimuth:                    return AZIMUTH_SIZE;
      case Slope_doppler_correction:   return SLOPE_DOPPLER_C_SIZE;
      case Satellite_latitude:         return LAT_SAT_SIZE;
      case Satellite_longitude:        return LON_SAT_SIZE;
      case Ambiguity:                  return AMBIGUITY_SIZE;
      case MSS_model:                  return MSS_MODE_SIZE;
      case Geoid_model:                return GEOID_MODE_SIZE;
      case ODLE_model:                 return ODLE_MODE_SIZE;
      case DEM_elevation:              return DEM_ELEV_SIZE;
      case DEM_id:                     return DEM_ID_SIZE;
      //
      case Dry_correction:             return DRY_C_SIZE;
      case Wet_correction:             return WET_C_SIZE;
      case IB_correction:              return IB_C_SIZE;
      case DAC_correction:             return DAC_C_SIZE;
      case Ionospheric_GIM_correction:   return IONO_GIM_SIZE;
      case Ionospheric_model_correction: return IONO_MODE_SIZE;
      case Ocean_tide:                 return H_OT_SIZE;
      case LPE_Ocean_tide:             return H_LPEOT_SIZE;
      case Ocean_loading_tide:         return H_OLT_SIZE;
      case Solid_earth_tide:           return H_SET_SIZE;
      case Geocentric_polar_tide:      return H_GPT_SIZE;
      case Surface_type:               return SURF_TYPE_SIZE;
      case Correction_status:          return CORR_STATUS_SIZE;
      case Correction_error:           return CORR_ERROR_SIZE;
      case SS_Bias_correction:         return SSB_SIZE;
      case Doppler_rc:                 return DOPP_RC_SIZE;
      case TR_instrument_rc:           return TR_INST_RC_SIZE;
      case R_instrument_rc:            return R_INST_RC_SIZE;
      case TR_instrument_gain_correction: return TR_INST_GAIN_C_SIZE;
      case R_instrument_gain_correction:  return R_INST_GAIN_C_SIZE;
      case Internal_phase_correction:     return INT_PHASE_C_SIZE;
      case External_phase_correction:     return EXT_PHASE_C_SIZE;
      case Noise_power_measurement:       return NOISE_PWR_SIZE;
      case Phase_slope_correction:        return PHASE_SLOPE_C_SIZE;
      //
      default: 
           assert(!"Field unknown!");     return 0;
    }
}

/*-- obtain fields properties --*/
FP
getProperties(int nd, int typenum, int typesize, npy_intp* ss)
{
    int j;
    FP fp;

    fp.typenum = typenum;
    fp.typesize = typesize;
    for (j = 0; j < nd; ++j) {
        fp.strides[j] = ss[j];
    }

    return fp;
} /* getProperties */

/*-- determine the total number of records in each field --*/
long int 
howManyRecs(t_cs_filehandle fH, BASELINE fBaseline)
{
    long int  n_records;

    // MSSL I/O librarry
    iTestEndian(); // test endianess

    if( fH != NULL )
    {
        // MSSL I/O librarry
        n_records = jCSNumRecordsInDataset( fH, 0 );
        return n_records;
    }
    else
    {
        printf( "Unable to open file.\n" );
        return EXIT_FAILURE;
    }

} /*howManyRecs*/

/*-- obtain the field data --*/
L2IData* 
csarray( t_cs_filehandle fH, long int n_records ) 
{

    L2IData*  fDataL2 = NULL;
    // MSSL I/O librarry
    fDataL2 = ptCSGetL2I( fH, 0, n_records, 0, NULL );

    if( fDataL2 != NULL )
    {
       /*
       Debug
       */
       //printf( "L2 day [0] = %"PRId32"\n", fDataL2[ 0 ].j_Day );
       //printf( "L2 day [1] = %"PRId32"\n", fDataL2[ 1 ].j_Day );

       return fDataL2;

    }
    else
    {
        printf( "Unable to read from file.\n" );
        return NULL;
    }

} /* csarray */

