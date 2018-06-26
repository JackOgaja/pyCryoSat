
=========
Overview
=========

PyCryoSat: Python c-extension module for reading European Space Agency's 
CryoSat-2 satellite In-depth L2 Measurement Data Set Records (L2I MDS) - 
https://earth.esa.int/web/eoportal/satellite-missions/c-missions/cryosat-2.
This extension uses an I/O library prepared by the software team at
Mullard Space Science Laboratory, UCL, London  

Before install:

*Download and install mssl-cryosat I/O library
from `European Space Agency's website <https://earth.esa.int/web/guest/software-tools/-/article/software-routines-7114>`_

To install:
::
    ~$ git clone https://github.com/JackOgaja/PyCryoSat.git
    ~$ cd PyCryoSat
    ~$ python setup.py build_ext --inplace

License:
========
   :The I/O library:  
   Copyright UCL/MSSL
    mssl-cryosat I/O software is developed by software team at  
    Mullard Space Science Laboratory, UCL, London.  
    For copyright and licensing information, 
    visit: http://cryosat.mssl.ucl.ac.uk

   :The python c-extension:  
   MIT License   
    For detailed copyright and licensing information, refer to the
    license file `LICENSE.md` in the top level directory.

