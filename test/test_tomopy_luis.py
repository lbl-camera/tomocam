# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:08:19 2016

@author: lbluque
"""

import multiprocessing as mp
import os
import tomopy
import time
# import astra
# import sirtfbp

datasets ={ '/home/svvenkatakrishnan/data/20130807_234356_OIM121R_SAXS_5x.h5': 1272}
algorithms = ('fbp',)
sinos = (10,10,1)
cores = (1,) #range(mp.cpu_count(), 0, -4)
f = open('benchmark_results.txt', 'a')

for dataset in datasets:
    f.write('*****************************************************************************************************\n')
    f.write(dataset + '\n\n')
    for algorithm in algorithms:
        for sino in sinos:
            for core in cores:
                start_time = time.time()
                tomo, flats, darks, floc = tomopy.read_als_832h5(dataset,
                                                                 sino=(0, sino, 1))
                end_time = time.time() - start_time
                f.write('Function: {0}, Number of sinos: {1}, Runtime (s): {2}\n'.format('read', sino, end_time))
                theta = tomopy.angles(tomo.shape[0])
                tomo = tomopy.normalize(tomo, flats, darks, ncore=core)
                end_time = time.time() - start_time - end_time 
                f.write('Function: {0}, Number of sinos: {1}, Number of cores: {2}, Runtime (s): {3}\n'.format('normalize', sino, core, end_time))
                tomo = tomopy.remove_stripe_fw(tomo, ncore=core)
                end_time = time.time() - start_time - end_time 
                f.write('Function: {0}, Number of sinos: {1}, Number of cores: {2}, Runtime (s): {3}\n'.format('stripe_fw', sino, core, end_time))
                rec = tomopy.recon(tomo, theta, center=datasets[dataset],
                                   algorithm=algorithm, emission=False,
                                   ncore=core)
                end_time = time.time() - start_time - end_time
                rec = tomopy.circ_mask(rec, 0)
                f.write('Function: {0}, Number of sinos: {1}, Number of cores: {2}, Algorithm: {3}, Runtime (s): {4}\n'.format('recon', sino, core, algorithm, end_time))
                outname = os.path.join('.', '{0}_{1}_slices_{2}_cores_{3}'.format(dataset.split('.')[0], str(algorithm), str(sino), str(core)), dataset.split('.')[0])
                tomopy.write_tiff_stack(rec, fname=outname)
                end_time = time.time() - start_time - end_time  
                f.write('Function: {0}, Number of images: {1}, Runtime (s): {2}\n\n'.format('write', rec.shape[0], end_time))
                f.flush()
f.close()
