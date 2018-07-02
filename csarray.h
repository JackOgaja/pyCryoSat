
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
   Day                      = 1,
   Sec                      = 2,
   Micsec                   = 3,
   USO_Correction_factor    = 4,
   Mode_id                  = 5, 
   Src_Seq_Counter          = 6,
   Instrument_config        = 7,
   Rec_Counter              = 8,
   Lat                      = 9,
   Lon                      = 10,
   Altitude                 = 11,
   Altitude_rate            = 12,
   Satellite_velocity       = 13,//Satellite_velocity[ 3 ],  
   Real_beam                = 14, //Real_beam[ 3 ],  
   Baseline                 = 15, //Baseline[ 3 ],  
   Star_Tracker_id          = 16,
   Roll_angle               = 17,
   Pitch_angle              = 18,
   Yaw_angle                = 19,
   Level2_MCD               = 20,
   /*-- Measurements group --*/
   Height                   = 21,
   Height_2                 = 22,
   Height_3                 = 23, 
   Sigma_0                  = 24,
   Sigma_0_2                = 25,
   Sigma_0_3                = 26,
   Significant_Wave_Height  = 27,
   Peakiness                = 28,
   Retracked_range_correction   = 29,
   Retracked_range_correction_2 = 30,
   Retracked_range_correction_3 = 31,
   Retracked_sig0_correction    = 32,
   Retracked_sig0_correction_2  = 33, 
   Retracked_sig0_correction_3  = 34,
   Retracked_quality_1       = 35,
   Retracked_quality_2       = 36,  
   Retracked_quality_3       = 37,    
   Retracked_3               = 38,
   Retracked_4               = 39,
   Retracked_5               = 40,
   Retracked_6               = 41,
   Retracked_7               = 42,
   Retracked_8               = 43,
   Retracked_9               = 44,
   Retracked_10              = 45,
   Retracked_11              = 46,
   Retracked_12              = 47,
   Retracked_13              = 48,
   Retracked_14              = 49,
   Retracked_15              = 50,
   Retracked_16              = 51,
   Retracked_17              = 52,
   Retracked_18              = 53,
   Retracked_19              = 54,
   Retracked_20              = 55,
   Retracked_21              = 56,
   Retracked_22              = 57,
   Retracked_23              = 58,
   Retracked_24              = 59,
   Retracked_25              = 60,
   Retracked_26              = 61,
   Power_echo_shape          = 62,
   Beam_behaviour_param      = 63, //Beam_behaviour_param[ 50 ],  
   Cross_track_angle         = 64,
   Cross_track_angle_correction  = 65,
   Coherence                 = 66,
   Interpolated_Ocean_Height = 67,
   Freeboard                 = 68,
   Surface_Height_Anomaly    = 69,
   Interpolated_sea_surface_h_anom   = 70,
   Ocean_height_interpolation_error  = 71,
   Interpolated_forward_records      = 72,
   Interpolated_backward_records     = 73,
   Time_interpolated_fwd_recs        = 74,
   Time_interpolated_bkwd_recs       = 75,
   Interpolation_error_flag          = 76,
   Measurement_mode                  = 77,
   Quality_flags             = 78 ,
   Retracker_flags           = 79,
   Height_status             = 80,
   Freeboard_status          = 81,
   Average_echoes            = 82,
   Wind_speed                = 83,
   //Spares1                   = 84, //Spares1[ 12 ],  
   /*-- Auxiliary Measurements group --*/
   Ice_concentration         = 84,
   Snow_depth                = 85,
   Snow_density              = 86,
   Discriminator             = 87,
   SARin_discriminator_1     = 88,
   SARin_discriminator_2     = 89,
   SARin_discriminator_3     = 90,
   SARin_discriminator_4     = 91,
   SARin_discriminator_5     = 92,
   SARin_discriminator_6     = 93,
   SARin_discriminator_7     = 94,
   SARin_discriminator_8     = 95,
   SARin_discriminator_9     = 96,
   SARin_discriminator_10    = 97,
   Discriminator_flags       = 98, 
   Attitude                  = 99,
   Azimuth                   = 100,
   Slope_doppler_correction  = 101,
   Satellite_latitude        = 102,
   Satellite_longitude       = 103,
   Ambiguity                 = 104,
   MSS_model                 = 105,
   Geoid_model               = 106,
   ODLE_model                = 107,   
   DEM_elevation             = 108, 
   DEM_id                    = 109,
   //Spares2, //Spares2[ 16 ],  
   /*-- External Corrections group --*/
   Dry_correction            = 110, 
   Wet_correction            = 111, 
   IB_correction             = 112, 
   DAC_correction            = 113, 
   Ionospheric_GIM_correction      = 114,
   Ionospheric_model_correction    = 115, 
   Ocean_tide                = 116,  
   LPE_Ocean_tide            = 117, 
   Ocean_loading_tide        = 118, 
   Solid_earth_tide          = 119, 
   Geocentric_polar_tide     = 120,
   Surface_type              = 121,
   Correction_status         = 122,
   Correction_error          = 123, 
   SS_Bias_correction        = 124,
   //Spares3,//Spares3[ 8 ],  
   /*-- Internal Corrections group --*/
   Doppler_rc                = 125,  
   TR_instrument_rc          = 126,
   R_instrument_rc           = 127,
   TR_instrument_gain_correction   = 128,  
   R_instrument_gain_correction    = 129, 
   Internal_phase_correction       = 130, 
   External_phase_correction       = 131, 
   Noise_power_measurement         = 132,
   Phase_slope_correction          = 133, 
   //Spares4 //Spares4[ 8 ] 
} FIELDS;

// functions prototypes
uint8_t fieldSize( FIELDS field ); 
long int howManyRecs( t_cs_filehandle fH, BASELINE fBaseline );
field_properties getProperties( int nd, int n, int s, npy_intp* ss );
L2IData* csarray( t_cs_filehandle fH, long int n_records );

#endif // #ifndef _CSARRAY_H_

