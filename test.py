#!/usr/bin/env python3

import numpy as np
import csarray as csa 

inFile = '/Users/jack/Arbeit/lynch_lab/data_processing/cryosat/codes/data/example/CS_OFFL_SIR_SARI2__20170131T223552_20170131T223747_C001.DBL'

print('The input file is : {}'.format(inFile))

l2ia = csa.l2Iarray(inFile, 4, 1)

np.set_printoptions(threshold=np.nan)
#print('L2IA = {}'.format(l2ia))
print(l2ia[2167])
print(l2ia[:])


