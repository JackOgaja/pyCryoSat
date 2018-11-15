# Copyright (C) 2018 Jack Ogaja.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# -*- coding: utf-8 -*-

__version__ = '0.1.0'
__description__ = 'pycryosat'
__author__ = 'Jack Ogaja  <jack.ogaja@gmail.com> '
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
    : outfile: output file name
    : readToDict: read data to a dictionary
    : readToCsvFile: read data to a csv file

    Reads .dbl/.hdr input files
    """

    outfile = 'pyCsOut.csv'

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
    1,    1,    1,     1,   1,      1,     1,    1,   
    1e-7, 1e-7, 1e-3, 1e-3, 1/1e3, 1/1e6, 1/1e6, 1e-7, 1e-7, 1e-7,
    1e-3, 1e-3, 1e-3, 1/100, 1/100, 1/100, 1e-3, 1e-2, 1e-3, 1e-3,
    1e-3, 1e-2, 1e-2, 1e-2, 1e-2,   1,     1,    1,    1,    1,  
    1,    1,    1,     1,   1,      1,     1,    1,    1,    1, 
    1,    1,    1,     1,   1,      1,     1,    1,    1,    1, 
    1,    1,    1,     1,   1,      1,     1,    1,    1,    1, 
    1e-3, 1e-2, 1,     1,   1,      1,     1,    1,    1,    1, 
    1,    1,    1,     1,   1,      1,     1,    1,    1,    1, 
    1,    1,    1,     1,   1,      1,     1,    1,    1,    1, 
    1,    1,    1,     1,   1,      1,     1,    1,    1,    1, 
    1,    1,    1,     1,   1,      1,     1,    1,    1,    1, 
    1,    1,    1,     1,   1,      1,     1,    1,    1,    1, 
    1,    1,     ]  

    def __init__(self, file_input, baseline=4):
        """
        create the class instance
        """
        self.input = file_input
        self.base = baseline

    @property
    def fields(self):
#        return self.__FieldsStrings_l2i
        return list(self.__FieldsStrings_l2i)
        
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

    def data_iter(self):


        df = self.__readFile()

        # consider the 2-D fields e.g. Satellite velocity, Real_beam, Baseline, and Beam_parameter
        rd = df[0:12]
        rdx = df[12][0]; rdy = df[12][1]; rdz = df[12][2] 
        rdr1 = df[13][0]; rdr2 = df[13][1]; rdr3 = df[13][2]  
        rdb1 = df[14][0]; rdb2 = df[14][1]; rdb3 = df[14][2]  
        rdp1 = df[59][0]; rdp2 = df[59][1]; rdp3 = df[59][2]; rdp4 = df[59][3]; rdp5 = df[59][4] 
        rdp6 = df[59][5]; rdp7 = df[59][6]; rdp8 = df[59][7]; rdp9 = df[59][8]; rdp10 = df[59][9] 
        rdp11 = df[59][10]; rdp12 = df[59][11]; rdp13 = df[59][12]; rdp14 = df[59][13]; rdp15 = df[59][14] 
        rdp16 = df[59][15]; rdp17 = df[59][16]; rdp18 = df[59][17]; rdp19 = df[59][18]; rdp20 = df[59][19] 
        rdp21 = df[59][20]; rdp22 = df[59][21]; rdp23 = df[59][22]; rdp24 = df[59][23]; rdp25 = df[59][24] 
        rdp26 = df[59][25]; rdp27 = df[59][26]; rdp28 = df[59][27]; rdp29 = df[59][28]; rdp30 = df[59][29] 
        rdp31 = df[59][30]; rdp32 = df[59][31]; rdp33 = df[59][32]; rdp34 = df[59][33]; rdp35 = df[59][34] 
        rdp36 = df[59][35]; rdp37 = df[59][36]; rdp38 = df[59][37]; rdp39 = df[59][38]; rdp40 = df[59][39] 
        rdp41 = df[59][40]; rdp42 = df[59][41]; rdp43 = df[59][42]; rdp44 = df[59][43]; rdp45 = df[59][44] 
        rdp46 = df[59][45]; rdp47 = df[59][46]; rdp48 = df[59][47]; rdp49 = df[59][48]; rdp50 = df[59][49] 

        yield  df[0], df[1], df[2], df[3], df[4], df[5], df[6], df[7], df[8], df[9], \
               df[10], df[11], rdx, rdy, rdz, rdr1, rdr2, rdr3, rdb1, rdb2, rdb3,    \
               df[15], df[16], df[17], df[18], df[19], df[20], df[21], df[22], df[23], df[24], \
               df[25], df[26], df[27], df[28], df[29], df[30], df[31], df[32], df[33], df[34], \
               df[35], df[36], df[37], df[38], df[39], df[40], df[41], df[42], df[43], df[44], \
               df[45], df[46], df[47], df[48], df[49], df[50], df[51], df[52], df[53], df[54], \
               df[55], df[56], df[57], df[58],                                                 \
               rdp1, rdp2, rdp3, rdp4, rdp5, rdp6, rdp7, rdp8, rdp9, rdp10,                    \
               rdp11, rdp12, rdp13, rdp14, rdp15, rdp16, rdp17, rdp18, rdp19, rdp20,           \
              rdp21, rdp22, rdp23, rdp24, rdp25, rdp26, rdp27, rdp28, rdp29, rdp30,           \
               rdp31, rdp32, rdp33, rdp34, rdp35, rdp36, rdp37, rdp38, rdp39, rdp40,           \
               rdp41, rdp42, rdp43, rdp44, rdp45, rdp46, rdp47, rdp48, rdp49, rdp50,           \
              df[60], df[61], df[62], df[63], df[64], df[65], df[66], df[67], df[68], df[69], \
               df[70], df[71], df[72], df[73], df[74], df[75], df[76], df[77], df[78], df[79], \
               df[80], df[81], df[82], df[83], df[84], df[85], df[86], df[87], df[88], df[89], \
              df[90], df[91], df[92], df[93], df[94], df[95], df[96], df[97], df[98], df[99], \
               df[100], df[101], df[102], df[103], df[104], df[105], df[106], df[107], df[108], df[109],\
               df[110], df[111], df[112], df[113], df[114], df[115], df[116], df[117], df[118], df[119],\
               df[120], df[121], df[122], df[123], df[124], df[125], df[126], df[127], df[128], df[129] 

    @property
    def readToCsvFile(self):
        """
        Write out data to a file
       """

        fname = self.outfile 

        fn_sat = ['Satellite_Vx', 'Satellite_Vy', 'Satellite_Vz']
        fn_rb = ['Real_beam1', 'Real_beam2', 'Real_beam3']
        fn_bs = ['Baseline1', 'Baseline2', 'Baseline3']
        fn_bb = [
                 'Beam_b_param1', 'Beam_b_param2', 'Beam_b_param3', 'Beam_b_param4', 'Beam_b_param5'
                 'Beam_b_param6', 'Beam_b_param7', 'Beam_b_param8', 'Beam_b_param9', 'Beam_b_param10'
                 'Beam_b_param11', 'Beam_b_param12', 'Beam_b_param13', 'Beam_b_param14', 'Beam_b_param15'
                 'Beam_b_param16', 'Beam_b_param17', 'Beam_b_param18', 'Beam_b_param19', 'Beam_b_param20'
                 'Beam_b_param21', 'Beam_b_param22', 'Beam_b_param23', 'Beam_b_param24', 'Beam_b_param25'
                 'Beam_b_param26', 'Beam_b_param27', 'Beam_b_param28', 'Beam_b_param29', 'Beam_b_param30'
                 'Beam_b_param31', 'Beam_b_param32', 'Beam_b_param33', 'Beam_b_param34', 'Beam_b_param35'
                 'Beam_b_param36', 'Beam_b_param37', 'Beam_b_param38', 'Beam_b_param39', 'Beam_b_param40'
                 'Beam_b_param41', 'Beam_b_param42', 'Beam_b_param43', 'Beam_b_param44', 'Beam_b_param45'
                 'Beam_b_param46', 'Beam_b_param47', 'Beam_b_param48', 'Beam_b_param49', 'Beam_b_param50'
                ]

        fn   =      self.__FieldsStrings_l2i[:12] \
                                         + fn_sat \
                                          + fn_rb \
                                          + fn_bs \
                + self.__FieldsStrings_l2i[15:59] \
                                          + fn_bb \
                  + self.__FieldsStrings_l2i[60:] 

        #- python 3X.
        data = self.data_iter()

        try:
              import csv
              try:
                  with open(fname, 'w') as f:
                        out_file = csv.DictWriter(f, fieldnames=fn, delimiter=';')
                        out_file.writeheader()
                        for rw in data:
                             out_file.writerow({
                                                fn[0]: rw[0], fn[1]: rw[1], fn[2]: rw[2],
                                                fn[3]: rw[3], fn[4]: rw[4], fn[5]: rw[5],
                                                fn[6]: rw[6], fn[7]: rw[7], fn[8]: rw[8],
                                                fn[9]: rw[9], fn[10]: rw[10], fn[11]: rw[11],
                                                fn[12]: rw[12], fn[13]: rw[13], fn[14]: rw[14],
                                                fn[15]: rw[15], fn[16]: rw[16], fn[17]: rw[17],
                                                fn[18]: rw[18], fn[19]: rw[19], fn[20]: rw[20],
                                                fn[21]: rw[21], fn[22]: rw[22], fn[23]: rw[23],
                                                fn[24]: rw[24], fn[25]: rw[25], fn[26]: rw[26],
                                                fn[27]: rw[27], fn[28]: rw[28], fn[29]: rw[29],
                                                fn[30]: rw[30], fn[31]: rw[31], fn[32]: rw[32],
                                                fn[33]: rw[33], fn[34]: rw[34], fn[35]: rw[35],
                                                fn[36]: rw[36], fn[37]: rw[37], fn[38]: rw[38],
                                                fn[39]: rw[39], fn[40]: rw[40], fn[41]: rw[41],
                                                fn[42]: rw[42], fn[43]: rw[43], fn[44]: rw[44],
                                                fn[45]: rw[45], fn[46]: rw[46], fn[47]: rw[47],
                                                fn[48]: rw[48], fn[49]: rw[49], fn[50]: rw[50],
                                                fn[51]: rw[51], fn[52]: rw[52], fn[53]: rw[53],
                                                fn[54]: rw[54], fn[55]: rw[55], fn[56]: rw[56],
                                                fn[57]: rw[57], fn[58]: rw[58], fn[59]: rw[59],
                                                fn[60]: rw[60], fn[61]: rw[61], fn[62]: rw[62],
                                                fn[63]: rw[63], fn[64]: rw[64], fn[65]: rw[65],
                                                fn[66]: rw[66], fn[67]: rw[67], fn[68]: rw[68],
                                                fn[69]: rw[69], fn[70]: rw[70], fn[71]: rw[71],
                                                fn[72]: rw[72], fn[73]: rw[73], fn[74]: rw[74],
                                                fn[75]: rw[75], fn[76]: rw[76], fn[77]: rw[77],
                                                fn[78]: rw[78], fn[79]: rw[79], fn[80]: rw[80],
                                                fn[81]: rw[81], fn[82]: rw[82], fn[83]: rw[83],
                                                fn[84]: rw[84], fn[85]: rw[85], fn[86]: rw[86],
                                                fn[87]: rw[87], fn[88]: rw[88], fn[89]: rw[89],
                                                fn[90]: rw[90], fn[91]: rw[91], fn[92]: rw[92],
                                                fn[93]: rw[93], fn[94]: rw[94], fn[95]: rw[95],
                                                fn[96]: rw[96], fn[97]: rw[97], fn[98]: rw[98],
                                                fn[99]: rw[99], fn[100]: rw[100], fn[101]: rw[101],
                                                fn[102]: rw[102], fn[103]: rw[103], fn[104]: rw[104],
                                                fn[105]: rw[105], fn[106]: rw[106], fn[107]: rw[107],
                                                fn[108]: rw[108], fn[109]: rw[109], fn[110]: rw[110],
                                                fn[111]: rw[111], fn[112]: rw[112], fn[113]: rw[113],
                                                fn[114]: rw[114], fn[115]: rw[115], fn[116]: rw[116],
                                                fn[117]: rw[117], fn[118]: rw[118], fn[119]: rw[119],
                                                fn[120]: rw[120], fn[121]: rw[121], fn[122]: rw[122],
                                                fn[123]: rw[123], fn[124]: rw[124], fn[125]: rw[125],
                                                fn[126]: rw[126], fn[127]: rw[127], fn[128]: rw[128],
                                                fn[129]: rw[129]
                                                })

                  print(' The output file {} has been written'.format(fname))

              except csv.Error as er:
                   sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, er))

        except ImportError:
               raise ImportError('cannot import csv module')


