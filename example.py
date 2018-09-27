#!/usr/bin/env python3.5

import numpy as np
from pycryosat import pycryosat as pc 

## See TODO file for optimization

inFile = '<path_to_.DBL_file>' # locate the input fsile
baseline = 4 # see the library documentation 
             # for different baselines definitions

print('The input file is : {}'.format(inFile))

# create an instance for pycryosat
rdata = pc(inFile, baseline)

# read data to a python dictionary
rd = rdata.readToDict

np.set_printoptions(threshold=np.nan)

# test whether the dictionary has been written correctly
for key, value in rd.items():
    print(key)
    print(value.shape)

print(rd['Freeboard'])
print(rd['Satellite_velocity'].shape)
[ni,nj] = rd['Satellite_velocity'].shape

print(' NI = {}'.format(ni))
print(' NJ = {}'.format(nj))

# OR write the data to a csv file
# The default csv file name is "pyCsOut.csv"
# The csv file name can be changed by specifying a name
#  for the attribute .outfile

rdata.outfile = 'testOut.csv'

rdata.readToCsvFile

