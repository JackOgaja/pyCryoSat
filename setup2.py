from distutils.core import setup, Extension

rootDir = '/Users/jack/Arbeit/lynch_lab/data_processing/cryosat/codes/c_idl/cs_tools_2_3/lib'
incDirs = [rootDir+'/mssl_shared',
           rootDir+'/mssl_cryosat']
libDirs = [rootDir+'/mssl_shared',
           rootDir+'/mssl_cryosat']

cryoSat_ext = Extension(
                         "pyCryoSatIO", 
                         sources = ["pyCryoSatIO_jack20180605.c", 
                                     "buffer.c"],
                         include_dirs = incDirs,
                         libraries = ['mssl_cryosat','mssl_shared', 'm'],
                         library_dirs = libDirs
                         )

setup (
       name = 'cryoSatIO',
       version = '0.1.0',
       ext_modules = [cryoSat_ext]
      )

