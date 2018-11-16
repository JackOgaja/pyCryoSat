
=========
PyCryoSat
=========

*Overview:*

PyCryoSat is a python C-extension and CUDA module for reading European Space Agency's 
CryoSat-2 satellite In-depth L2 Measurement Data Set Records (L2I MDS) - 
https://earth.esa.int/web/eoportal/satellite-missions/c-missions/cryosat-2.
This extension uses an I/O library prepared by the software team at
Mullard Space Science Laboratory, UCL, London  

*The program layout:*
::
  .../pyCryoSat/  
  |- README  
  |- LICENSE  
  |- setup.py  
  |- src  
  |    |- csarray.c 
  |    |- pycsarray.c 
  |- cuda  
  |    |- d_csGetL2I.cu  
  |- include  
  |    |- csarray.h  
  |- lib  
  |    |- __init__.py 
  |- pycryosat.py  
  |- example.py  

*Dependencies*
    - python3.5
    - mssl-cryosat: http://cryosat.mssl.ucl.ac.uk

*Before installation:*

-Download and install mssl-cryosat I/O library
from `European Space Agency's website <https://earth.esa.int/web/guest/software-tools/-/article/software-routines-7114>`_

*To install:*
::
    ~$ git clone https://github.com/JackOgaja/PyCryoSat.git
    ~$ cd PyCryoSat
    ~$ python3.5 setup.py build_ext -b lib

Usage:
::
    ~$ ./example.py 

License:
========

   MIT License   
    For detailed copyright and licensing information, refer to the
    license file `LICENSE.md` in the top level directory.

