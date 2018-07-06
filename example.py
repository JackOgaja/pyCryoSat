#!/usr/bin/env python3

import numpy as np
import lib.csarray as csa 

inFile = '<path_to_.DBL_file>' # locate the input file

print('The input file is : {}'.format(inFile))

l2ia = csa.l2Iarray(inFile, 4, 1)

np.set_printoptions(threshold=np.nan)
#print('L2IA = {}'.format(l2ia))
print(l2ia[2167])
print(l2ia[:])


