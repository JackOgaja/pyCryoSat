from distutils.core import setup, Extension
import numpy as np

msslDir = '/Users/jack/Arbeit/lynch_lab/data_processing/cryosat/codes/c_idl/cs_tools_2_3/lib'
incDirs = [msslDir+'/mssl_shared',
           msslDir+'/mssl_cryosat']
libDirs = [msslDir+'/mssl_shared',
           msslDir+'/mssl_cryosat']

cryoSat_ext = Extension(
                         "csarray", 
                         sources = ["pycsarray.c", 
                                     "csarray.c"],
                         include_dirs = incDirs,
                         libraries = ['mssl_cryosat','mssl_shared', 'm'],
                         library_dirs = libDirs,
                         extra_compile_args = ['-Wall']
                         )

setup (
       name = 'cryoSatArray',
       version = '0.1.0',
       ext_modules = [cryoSat_ext],
       include_dirs = [np.get_include()]
      )

