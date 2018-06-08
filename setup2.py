from distutils.core import setup, Extension

msslDir = '/Users/jack/Arbeit/lynch_lab/data_processing/cryosat/codes/c_idl/cs_tools_2_3/lib'
incDirs = [msslDir+'/mssl_shared',
           msslDir+'/mssl_cryosat']
libDirs = [msslDir+'/mssl_shared',
           msslDir+'/mssl_cryosat']

cryoSat_ext = Extension(
                         "pyCryoSatIO", 
                         sources = ["pyCryoSatIO_jack20180605.c", 
                                     "csarray.c"],
                         include_dirs = incDirs,
                         libraries = ['mssl_cryosat','mssl_shared', 'm'],
                         library_dirs = libDirs
                         )

setup (
       name = 'cryoSatIO',
       version = '0.1.0',
       ext_modules = [cryoSat_ext]
      )

