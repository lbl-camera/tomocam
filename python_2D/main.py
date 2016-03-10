import matplotlib.pyplot as plt
import tomopy
from normalize import normalize_bo
import argparse
from XT_argsParser import bl832inputs_parser
import time
import os
import math
import numpy as np


def main():

        parser = argparse.ArgumentParser()
        inputs = bl832inputs_parser(parser)

        algorithm='fbp'
        tomo, flats, darks, floc = tomopy.read_als_832h5(inputs['input_hdf5'],ind_tomo=range(1,inputs['num_views'],inputs['view_subsmpl_fact']),sino=(inputs['z_start'], inputs['z_start']+inputs['z_numElts'], 1))
        print('Data read complete')
        print tomo.shape

        print('Displaying  sinogram')
#        imgplot = plt.imshow(tomo[:,0,:])

        print('Generating angles')
        theta = tomopy.angles(tomo.shape[0])

        print('Normalization')
        tomo = normalize_bo(tomo, flats, darks,inputs['num_dark'])

        print('Ring removal')
        tomo = tomopy.remove_stripe_fw(tomo)

        print('Recon')
        rec = tomopy.recon(tomo, theta, center=1294,algorithm=algorithm,emission=False)

        print('Masking')
        rec = tomopy.circ_mask(rec, 0)

        tomopy.write_tiff_stack(rec, 'test.tiff')
        print 'main: Done!'

main()


