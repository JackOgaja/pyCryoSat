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

        ddict = {'Day':l2ia[1]}
        ddict['Sec'] = l2ia[2]
        ddict['Micsec'] = l2ia[3]
        ddict['USO_Correction_factor'] = l2ia[4]
        ddict['Mode_id'] = l2ia[5]
        ddict['Src_Seq_Counter'] = l2ia[6]
        ddict['Instrument_config'] = l2ia[7]

        ddict['Satellite_velocity'] = l2ia[13]

        return ddict 

