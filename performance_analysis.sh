#!/bin/sh

# profile the code for optimization measurements
#python3.5 -m cProfile -s cumtime -o tstats.out example2.py
python3.5 -m cProfile -s cumtime example2.py

# process the measurements into ascii 
#./performance_stats.py > s_stats.tx

