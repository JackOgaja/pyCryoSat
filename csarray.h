
#ifndef _CSARRAY_H_
#define _CSARRAY_H_

/*
The module interface
*/

// SYSTEM INTERFACE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

// NUMPY INTERFACE
#include <numpy/arrayobject.h>

// MSSL I/O LIB INTERFACE
#include "ptCSGetFileHandle.h"
#include "vCSFreeFileHandle.h"
#include "ptCSGetL2I.h"
#include "jCSNumRecordsInDataset.h"
#include "jIOFunctions.h"
#include "CS_Defines.h"

/*
SOME TEST
*/
#define CONCAT(A,B)         A ## B
#define EXPAND_CONCAT(A,B)  CONCAT(A, B)

#define ARGN(N, LIST)       EXPAND_CONCAT(ARG_, N) LIST
#define ARG_0(A0, ...)      A0
#define ARG_1(A0, A1, ...)  A1
#define ARG_2(A0, A1, A2, ...)      A2
#define ARG_3(A0, A1, A2, A3, ...)  A3
#define ARG_4(A0, A1, A2, A3, A4, ...)      A4
#define ARG_5(A0, A1, A2, A3, A4, A5, ...)  A5
#define ARG_6(A0, A1, A2, A3, A4, A5, A6, ...)      A6
#define ARG_7(A0, A1, A2, A3, A4, A5, A6, A7, ...)  A7
#define ARG_8(A0, A1, A2, A3, A4, A5, A6, A7, A8, ...)      A8
#define ARG_9(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, ...)  A9
#define ARG_10(A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, ...)    A10

/* define array of sizes */

#define SIZE_LIST_1 ( 2,  4,  8,  12,  16,   100)
#define SIZE_LIST_2 (11, 65, 222, 112, 444, 1000)

#define S1 ARGN(MODE, SIZE_LIST_1)
#define S2 ARGN(MODE, SIZE_LIST_2)

#define MODE 2 
/*
END SOME TEST
*/
// field sizes in bytes
/*-- time and orbit group --*/
#define DAY_SIZE (4)
#define SEC_SIZE (4)
#define MICSEC_SIZE (4)
#define USO_CORR_SIZE (4)
#define MODE_ID_SIZE (4)
#define SSC_SIZE (4)
#define INST_CONFIG_SIZE (4)
#define REC_COUNT_SIZE (4)
#define LAT_SIZE (4)
#define LON_SIZE (4)
#define ALT_SIZE (4)
#define ALT_RATE_SIZE (4)
#define SAT_VELOCITY_SIZE (4*3)
#define REAL_BEAM_SIZE (4*3)
#define BASELINE_SIZE (4*3)
#define ST_ID_SIZE (4)
#define ROLL_ANGLE_SIZE (4)
#define PITCH_ANGLE_SIZE (4)
#define YAW_ANGLE_SIZE (4)
#define L2_MCD_SIZE (4)
/*-- measurement group --*/
#define HEIGHT_SIZE (4)
#define HEIGHT_2_SIZE (4)
#define HEIGHT_3_SIZE (4)
#define SIG0_SIZE (4)
#define SIG0_2_SIZE (4)
#define SIG0_3_SIZE (4)
#define SWH_SIZE (4)
#define PEAKINESS_SIZE (4)
#define RETRK_RANGE_C_SIZE (4)
#define RETRK_RANGE_C_2_SIZE (4)
#define RETRK_RANGE_C_3_SIZE (4)
#define RETRK_SIG0_C_SIZE (4)
#define RETRK_SIG0_C_2_SIZE (4)
#define RETRK_SIG0_C_3_SIZE (4)
#define RETRK_QUALITY_1_SIZE (4)
#define RETRK_QUALITY_2_SIZE (4)
#define RETRK_QUALITY_3_SIZE (4)
#define RETRK_3_SIZE (4)
#define RETRK_4_SIZE (4)
#define RETRK_5_SIZE (4)
#define RETRK_6_SIZE (4)
#define RETRK_7_SIZE (4)
#define RETRK_8_SIZE (4)
#define RETRK_9_SIZE (4)
#define RETRK_10_SIZE (4)
#define RETRK_11_SIZE (4)
#define RETRK_12_SIZE (4)
#define RETRK_13_SIZE (4)
#define RETRK_14_SIZE (4)
#define RETRK_15_SIZE (4)
#define RETRK_16_SIZE (4)
#define RETRK_17_SIZE (4)
#define RETRK_18_SIZE (4)
#define RETRK_19_SIZE (4)
#define RETRK_20_SIZE (4)
#define RETRK_21_SIZE (4)
#define RETRK_22_SIZE (4)
#define RETRK_23_SIZE (4)
#define ECHO_SHAPE_SIZE (4)
#define BB_PARAM_SIZE (4*50)
#define XTRACK_ANGLE_SIZE (4)
#define XTRACK_ANGLE_C_SIZE (4)
#define COHERENCE_SIZE (4)
#define OCEAN_HT_SIZE (4)
#define FREEBOARD_SIZE (4)
#define SHA_SIZE (4)
#define SSHA_INTERP_SIZE (4)
#define INTERP_ERR_SIZE (4)
#define INTERP_CNT_FWD_SIZE (4)
#define INTERP_CNT_BKWD_SIZE (4)
#define INTERP_TIME_FWD_SIZE (4)
#define INTERP_TIME_BKWD_SIZE (4)
#define INTERP_ERROR_F_SIZE (4)
#define MEAS_MODE_SIZE (4)
#define QUALITY_F_SIZE (4)
#define RETRACKER_F_SIZE (4)
#define HT_STATUS_SIZE (4)
#define FREEB_STATUS_SIZE (4)
#define N_AVG_SIZE (4)
#define WIND_SPEED_SIZE (4)
#define SPARES1_SIZE (4*12)
/*-- Auxiliary measurement group --*/
#define ICE_CONC_SIZE (4)
#define SNOW_DEPTH_SIZE (4)
#define SNOW_DENSITY_SIZE (4)
#define DISCRIMINATOR_SIZE (4)
#define SARIN_DISC_1_SIZE (4)
#define SARIN_DISC_2_SIZE (4)
#define SARIN_DISC_3_SIZE (4)
#define SARIN_DISC_4_SIZE (4)
#define SARIN_DISC_5_SIZE (4)
#define SARIN_DISC_6_SIZE (4)
#define SARIN_DISC_7_SIZE (4)
#define SARIN_DISC_8_SIZE (4)
#define SARIN_DISC_9_SIZE (4)
#define SARIN_DISC_10_SIZE (4)
#define DISCRIM_F_SIZE (4)
#define ATTITUDE_SIZE (4)
#define AZIMUTH_SIZE (4)
#define SLOPE_DOPPLER_C_SIZE (4)
#define LAT_SAT_SIZE (4)
#define LON_SAT_SIZE (4)
#define AMBIGUITY_SIZE (4)
#define MSS_MODE_SIZE (4)
#define GEOID_MODE_SIZE (4)
#define ODLE_MODE_SIZE (4)
#define DEM_ELEV_SIZE (4)
#define DEM_ID_SIZE (4)
#define SPARES2_SIZE (4*16)
/*-- External Corrections group --*/
#define DRY_C_SIZE (4)
#define WET_C_SIZE (4)
#define IB_C_SIZE (4)
#define DAC_C_SIZE (4)
#define IONO_GIM_SIZE (4)
#define IONO_MODE_SIZE (4)
#define H_OT_SIZE (4)
#define H_LPEOT_SIZE (4)
#define H_OLT_SIZE (4)
#define H_SET_SIZE (4)
#define H_GPT_SIZE (4)
#define SURF_TYPE_SIZE (4)
#define CORR_STATUS_SIZE (4)
#define CORR_ERROR_SIZE (4)
#define SSB_SIZE (4)
#define SPARES3_SIZE (4*8)
/*-- Internal Corrections group --*/
#define DOPP_RC_SIZE (4)
#define TR_INST_RC_SIZE (4)
#define R_INST_RC_SIZE (4)
#define TR_INST_GAIN_C_SIZE (4)
#define R_INST_GAIN_C_SIZE (4)
#define INT_PHASE_C_SIZE (4)
#define EXT_PHASE_C_SIZE (4)
#define NOISE_PWR_SIZE (4)
#define PHASE_SLOPE_C_SIZE (4)
#define SPARES4 (4*8)

// New data types
typedef struct _field_properties {
   /*
    some unique properties of the data fields 
   */
   int typenum;
   /* the array data-type */
   int typesize;
   /* the field size in bytes */
   npy_intp strides[NPY_MAXDIMS];
   /* array strides */
} field_properties;

typedef struct _arrayStruct {
    /*
     construct the field arrays 
    */
    void* arrayData;
    /* field array data */
    field_properties fp;
    /* fields properties */
} arrayStruct;

/*
Select different data fields
*/
typedef enum _FIELDS
{
   /*-- Unknown field --*/
   Field_Unknown,
   /*-- Time and Orbit group --*/
   Day,
   Sec,
   Micsec,
   USO_Correction_factor,
   Mode_id, 
   Src_Seq_Counter,
   Instrument_config,
   Rec_Counter,
   Lat,
   Lon,
   Altitude,
   Altitude_rate,
   //Satellite_velocity[ 3 ],
   //Real_beam[ 3 ],
   //Baseline[ 3 ],
   Star_Tracker_id,
   Roll_angle,
   Pitch_angle,
   Yaw_angle,
   Level2_MCD,
   /*-- Measurements group --*/
   Height,
   Height_2,
   Height_3, 
   Sigma_0,
   Sigma_0_2,
   Sigma_0_3,
   Significant_Wave_Height,
   Peakiness,
   Retracked_range_correction,
   Retracked_range_correction_2,
   Retracked_range_correction_3,
   Retracked_sig0_correction,
   Retracked_sig0_correction_2, 
   Retracked_sig0_correction_3,
   Retracked_quality_1,
   Retracked_quality_2,  
   Retracked_quality_3,    
   Retracked_3,
   Retracked_4,
   Retracked_5,
   Retracked_6,
   Retracked_7,
   Retracked_8,
   Retracked_9,
   Retracked_10,
   Retracked_11,
   Retracked_12,
   Retracked_13,
   Retracked_14,
   Retracked_15,
   Retracked_16,
   Retracked_17,
   Retracked_18,
   Retracked_19,
   Retracked_20,
   Retracked_21,
   Retracked_22,
   Retracked_23,
   Retracked_24,
   Retracked_25,
   Retracked_26,
   Power_echo_shape,
   //Beam_behaviour_param[ 50 ],
   Cross_track_angle,
   Cross_track_angle_correction,
   Coherence,
   Interpolated_Ocean_Height,
   Freeboard,
   Surface_Height_Anomaly,
   Interpolated_sea_surface_h_anom,
   Ocean_height_interpolation_error,
   Interpolated_forward_records,
   Interpolated_backward_records,
   Time_interpolated_fwd_recs,
   Time_interpolated_bkwd_recs,
   Interpolation_error_flag,
   Measurement_mode,
   Quality_flags,
   Retracker_flags,
   Height_status,
   Freeboard_status,
   Average_echoes,
   Wind_speed,
   //Spares1[ 12 ],
   /*-- Auxiliary Measurements group --*/
   Ice_concentration,
   Snow_depth,
   Snow_density,
   Discriminator,
   SARin_discriminator_1,
   SARin_discriminator_2,
   SARin_discriminator_3,
   SARin_discriminator_4,
   SARin_discriminator_5,
   SARin_discriminator_6,
   SARin_discriminator_7,
   SARin_discriminator_8,
   SARin_discriminator_9,
   SARin_discriminator_10,
   Discriminator_flags, 
   Attitude,
   Azimuth,
   Slope_doppler_correction,
   Satellite_latitude,
   Satellite_longitude,
   Ambiguity,
   MSS_model,
   Geoid_model,
   ODLE_model,   
   DEM_elevation, 
   DEM_id,
   //Spares2[ 16 ],
   /*-- External Corrections group --*/
   Dry_correction, 
   Wet_correction, 
   IB_correction, 
   DAC_correction, 
   Ionospheric_GIM_correction,
   Ionospheric_model_correction, 
   Ocean_tide,  
   LPE_Ocean_tide, 
   Ocean_loading_tide, 
   Solid_earth_tide, 
   Geocentric_polar_tide,
   Surface_type,
   Correction_status,
   Correction_error, 
   SS_Bias_correction,
   //Spares3[ 8 ],
   /*-- Internal Corrections group --*/
   Doppler_rc,  
   TR_instrument_rc,
   R_instrument_rc,
   TR_instrument_gain_correction,  
   R_instrument_gain_correction, 
   Internal_phase_correction, 
   External_phase_correction, 
   Noise_power_measurement,
   Phase_slope_correction, 
   //Spares4[ 8 ]
} FIELDS;

// functions prototypes
uint8_t fieldSize( FIELDS field ); 
long int howManyRecs( t_cs_filehandle fH, BASELINE fBaseline );
field_properties getProperties( int n, int s );
L2IData* csarray( t_cs_filehandle fH, long int n_records );

#endif // #ifndef _CSARRAY_H_

