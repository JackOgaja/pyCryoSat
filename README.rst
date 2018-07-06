
=========
Overview
=========

PyCryoSat: Python C-extension module for reading European Space Agency's 
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
  |- include  
  |    |- csarray.h  
  |- lib  
  |    |- __init__.py 
  |- pycryosat  
  |    |- __init__.py 
  |    |- pycryosat.py  
  |- example.py  

*Before installation:*

-Download and install mssl-cryosat I/O library
from `European Space Agency's website <https://earth.esa.int/web/guest/software-tools/-/article/software-routines-7114>`_

*To install:*
::
    ~$ git clone https://github.com/JackOgaja/PyCryoSat.git
    ~$ cd PyCryoSat
    ~$ python setup.py build_ext -b lib

To test:
::
    ~$ python example.py 

License:
========
   :The I/O library:  
   Copyright UCL/MSSL
    mssl-cryosat I/O software is developed by software team at  
    Mullard Space Science Laboratory, UCL, London.  
    For copyright and licensing information, 
    visit: http://cryosat.mssl.ucl.ac.uk

   :The python C-extension:  
   MIT License   
    For detailed copyright and licensing information, refer to the
    license file `LICENSE.md` in the top level directory.

