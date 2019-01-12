#!/usr/bin/env python3.5

import pstats
perf = pstats.Stats('tstats.out')
perf.strip_dirs().sort_stats(-1).print_stats()

