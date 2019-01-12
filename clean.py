#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  Do some cleaning!
"""
try:
    import os
    from distutils.dir_util import remove_tree
    from shutil import copyfile
except ImportError:
    raise ImportError('cannot import necessary modules')

def cleanDir(dir):
    if os.path.exists(dir):
        print("Cleaning directory: " + dir + "\n")
        for f in os.listdir(dir):
            if not os.path.isdir(os.path.join(dir, f)) \
                    and not f.lower().endswith(".pyc") and not f.lower().endswith(".py"):
                os.remove(os.path.join(dir, f))
        for f in os.listdir(dir):
            if not os.path.isdir(f) and f.lower().endswith(".pyc"):
                copyfile(os.path.join(dir, f), os.path.join(dir, f[:-5]))

def cleanPycFiles(dir):
    print("Cleaning .pyc files")
    for root, dirs, files in os.walk(dir):
        for file in files:
            full_path = os.path.join(root, file)
            if full_path.lower().endswith(".pyc"):
                os.remove(full_path)

def cleanLibDir():
    print("Cleaning lib directory...\n")
    for root, dirs, files in os.walk("lib"):
        for file in files:
            full_path = os.path.join(root, file)
            if full_path.lower().endswith(".so"):
                os.remove(full_path)
        for dir in dirs:
            _cache_dir = os.path.join(root, dir)
            if _cache_dir.lower().endswith("__pycache__"):
                remove_tree(_cache_dir, verbose=1)

print("")
print("Starting clean.\n")

_Py_file_loc = os.path.dirname(os.path.realpath(__file__))
_sample_dir = os.path.join(_Py_file_loc, "sample")
_dist_dir = ""
_config_dir = ""

# Delete the distribution dir if it exists
if os.path.exists(_dist_dir):
    print("Removing dist directory: " + _dist_dir + "\n")
    remove_tree(_dist_dir, verbose=1)

# Clean lib dir
cleanLibDir()

# Clean the config dir
cleanDir(_config_dir)

# Clean the samples dir
cleanDir(_sample_dir)

# Clean .pyc files


